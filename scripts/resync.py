import ctypes
import multiprocessing as mp
import numpy as np
import os
import sys
import time
import wafel

sys.path.append("core")
import inputTools
import sm64Tools
from mpManager import ProcessGroup, printMP, barrierSync


# Thread count (should be slightly less than total CPU threads)
NUM_THREADS = 6

# WAFEL info
DLL_PATH = "libsm64"
GAME_VERSION = "us"

# Target m64s info
SOURCE_START_FRAME = 0 # Start of sync tunnel
SOURCE_END_FRAME = 0 # End of sync tunnel
SOURCE_NAME = ".m64"

RESYNC_START_FRAME = SOURCE_START_FRAME # Sync start in output
RESYNC_NAME = ".m64"
OUT_NAME = ".m64"

"""Source of camera inputs in output file (in resync range)
    "old" - use cam inputs from source
    "new" - use cam inputs from resync file
    None  - no cam inputs"""
CAM_INPUTS_SOURCE = "new"

# Basis m64 info (for state loaded)
STATE_OFFSET = 0
BASIS_NAME = "basis.m64"

# Input perturbing options
LOOKBACK_FRAMES = 30
PERT_PROB_SEARCH = 0.1
PERT_SIZE_SEARCH = 16
PERT_PROB_OPT = 0.05
PERT_SIZE_OPT = 4

# Simulated annealing options
DIST_LENIENCY = 0.2 # Max amount score can be worse than best and be used
INITIAL_TEMP = 0.1
TEMP_DECAY_RATE = 0.999

# Resync info
OVEREXTENDED_DIST = 100 # Dist where metric becomes unusable
MAX_RESYNC_DIST = 2 # Max acceptable dist
BRUTEFORCE_START_DIST = 0.2 # Starts wider bruteforce search above this
BRUTEFORCE_ACCEPT_DIST = 0.05 # Stops wider bruteforce search at this
INSTANT_ACCEPT_DIST = 0.01 # Stops current frame joystick search below this

JOYSTICK_SEARCH_RANGE = 128

# Weights for resync distance score
POS_WEIGHT = 1
VEL_WEIGHT = 0.1
HSPD_WEIGHT = 50
ANGLE_WEIGHT = 0.001
ANGLE_VEL_WEIGHT = 0.01
ACTION_PENALTY = 100000


# Run after basis m64 inputs, for fixing rng/global timer etc to match state loaded m64
def syncStateLoaded(game):
    pass

# Get game vars relevant to tunnel
def getGameVars(game):
    return {"pos": game.read("gMarioState.pos"),
            "vel": game.read("gMarioState.vel"),
            "hspd": game.read("gMarioState.forwardVel"),
            "face_angle": game.read("gMarioState.faceAngle"),
            "angle_vel": game.read("gMarioState.angleVel"),
            "num_coins": game.read("gMarioState.numCoins"),
            "action": game.read("gMarioState.action"),
            "cam_yaw": sm64Tools.getCamYaw(game),
            "stick_mag": game.read("gMarioState.controller->stickMag"),
            "intended_yaw": game.read("gMarioState.intendedYaw")}


# Squared dist prioritizing being in the same int
weightedFloatDist = lambda a, b: a - b + int(a) - int(b)

# Squared dist (modulo) prioritizing matching top 12 bits for trig
shortDist = lambda a, b: (int(a - b + 32768) & 65535) - 32768
weightedShortDist = lambda a, b: shortDist(a, b) + shortDist(a & 65520, b & 65520)

# Get distance score for resyncing
def stateDist(v1, v2):
    dist = HSPD_WEIGHT*((v1["hspd"]-v2["hspd"]) ** 2)
    for i in range(3):
        dist += POS_WEIGHT*((v1["pos"][i]-v2["pos"][i]) ** 2)
        dist += VEL_WEIGHT*((v1["vel"][i]-v2["vel"][i]) ** 2)
    for i in range(2):
        dist += ANGLE_WEIGHT*(weightedShortDist(v1["face_angle"][i], v2["face_angle"][i]) ** 2)
        dist += ANGLE_VEL_WEIGHT*(shortDist(v1["angle_vel"][i], v2["angle_vel"][i]) ** 2)
    if (v1["action"] != v2["action"]):
        dist += ACTION_PENALTY
    return dist


# Random input perturbation
def perturbInput(inp, frame, p=PERT_PROB_SEARCH, s=PERT_SIZE_SEARCH):
    stick_x, stick_y, buttons = inputTools.splitInputs(inp)
    if (np.random.random() < p):
        stick_x += np.random.randint(-s, s+1)
        stick_x &= 255
    if (np.random.random() < p):
        stick_y += np.random.randint(-s, s+1)
        stick_y &= 255
    if (np.random.random() < 0.1*p):
        stick_x = 0
        stick_y = 0
    return inputTools.joinInputs(stick_x, stick_y, buttons)


# Multiprocess bruteforcer for resyncing by bruteforcing large segments
def mpResyncBruteforce(thread_id, args):
    globals().update(args)

    # Load WAFEL
    lib_full_path = os.path.join(DLL_PATH, f"sm64_{GAME_VERSION}.dll")
    game = wafel.Game(lib_full_path)

    # Advance to start frame
    if STATE_LOADED:
        for frame in range(STATE_OFFSET):
            inputTools.advanceFrame(game, basis_inputs[frame])
        syncStateLoaded(game)
    for frame in range(RESYNC_START_FRAME):
        inputTools.advanceFrame(game, setup_inputs[frame])
    start_state = game.save_state()

    cur_inputs = np.ndarray(LENGTH, np.uint32)
    test_inputs = np.ndarray(LENGTH, np.uint32)

    setup_barrier.wait()
    if (thread_id == 0):
        with setup_completed.get_lock():
            setup_completed.value = True

    while True:
        # Wait for work
        while not bruteforce_active.value:
            time.sleep(0.1)

        # Copy setup
        cur_start = block_start.value
        cur_end = block_end.value
        cur_inputs[:cur_end] = new_inputs[:cur_end]
        test_inputs[:cur_end] = new_inputs[:cur_end]
        local_best_index = 0
        target_vars = tunnel[cur_end - 1]
        setup_joystick_i = last_sync_i.value
        target_joystick_i = cur_joystick_i.value
        cur_dist = best_dist.value
        sync_barrier.wait()

        # Advance to block start
        game.load_state(start_state)
        for i in range(cur_start):
            inputTools.advanceFrame(game, new_inputs[i])
        block_start_state = game.save_state()

        # Bruteforce loop
        temp = INITIAL_TEMP
        while bruteforce_active.value:
            cur_best_index = best_index.value
            if (cur_best_index != local_best_index):
                local_best_index = cur_best_index
                with best_inputs.get_lock():
                    cur_inputs[cur_start:cur_end] = best_inputs[cur_start:cur_end]
                cur_dist = best_dist.value

            game.load_state(block_start_state)
            for i in range(cur_start, cur_end):
                frame = i + RESYNC_START_FRAME

                # Maintain stability while bruteforcer prev dist
                if ((i == target_joystick_i) and (cur_dist >= OVEREXTENDED_DIST)):
                    new_input = new_inputs[i]
                else:
                    new_input = cur_inputs[i]

                if joystick_reqs[i]:
                    if (cur_dist >= MAX_RESYNC_DIST):
                        pert_prob, pert_size = PERT_PROB_SEARCH, PERT_SIZE_SEARCH
                    else:
                        pert_prob, pert_size = PERT_PROB_OPT, PERT_SIZE_OPT
                    new_input = perturbInput(new_input, frame, p=pert_prob, s=pert_size)
                test_inputs[i] = new_input
                inputTools.advanceFrame(game, new_input)
                
                if (i == setup_joystick_i):
                    setup_vars = getGameVars(game)

            """If current dist is very large (e.g. missing a bonk),
               the dist metric this step becomes meaningless
               Instead, if dist is above max resync dist, we optimize
               dist of last step until cur step is in range"""
            dist = stateDist(target_vars, getGameVars(game))
            using_current_dist = ((setup_joystick_i == -1) or (dist < OVEREXTENDED_DIST))
            if using_current_dist:
                proxy_dist = dist
            else:
                setup_dist = stateDist(tunnel[setup_joystick_i], setup_vars)
                proxy_dist = OVEREXTENDED_DIST + setup_dist

            # Check new best                
            with best_dist.get_lock():
                if (proxy_dist < best_dist.value):
                    # Update new best
                    best_dist.value = proxy_dist
                    local_best_index = best_index.value + 1
                    if using_current_dist:
                        printMP(f"New best dist: %.10f" % dist)
                    else:
                        printMP(f"New best prev dist: %.10f" % setup_dist)

                    with best_inputs.get_lock():
                        best_inputs[:] = test_inputs
                    with best_index.get_lock():
                        best_index.value = local_best_index
                    
                    test_inputs, cur_inputs = cur_inputs, test_inputs
                    cur_dist = proxy_dist
                elif ((proxy_dist + DIST_LENIENCY) > best_dist.value):
                    if (np.random.random() < np.exp((cur_dist - proxy_dist) / temp)):
                        test_inputs, cur_inputs = cur_inputs, test_inputs
                        cur_dist = proxy_dist

            temp *= TEMP_DECAY_RATE

        # Reset
        sync_barrier.wait()


# Resync m64 to match source as closely as possible
def resync():
    LENGTH = SOURCE_END_FRAME - SOURCE_START_FRAME
    
    # Setup base inputs
    new_inputs = source_inputs[SOURCE_START_FRAME:SOURCE_END_FRAME].copy()
    
    if (CAM_INPUTS_SOURCE == None):
        cam_src = "none"
    else:
        cam_src = CAM_INPUTS_SOURCE.lower()
        
    if (cam_src == "new"):
        new_inputs &= 0xFFE0FFFF
        max_available = min(LENGTH, resync_inputs.size - RESYNC_START_FRAME)
        new_inputs[:max_available] |= resync_inputs[RESYNC_START_FRAME:RESYNC_START_FRAME + max_available] & 0x1F0000
    elif (cam_src == "none"):
        new_inputs &= 0xFFE0FFFF
    elif (cam_src != "old"):
        raise ValueError("Invalid cam inputs source (must be \"old\", \"new\" or None\")")
    
    # Load WAFEL
    lib_full_path = os.path.join(DLL_PATH, f"sm64_{GAME_VERSION}.dll")
    game = wafel.Game(lib_full_path)

    # Advance to start frame
    if STATE_LOADED:
        for frame in range(STATE_OFFSET):
            inputTools.advanceFrame(game, basis_inputs[frame])
        syncStateLoaded(game)
    base_state = game.save_state()
    for frame in range(SOURCE_START_FRAME):
        inputTools.advanceFrame(game, source_inputs[frame])

    # Create tunnel and get sync requirements
    print("Creating tunnel...")
    joystick_reqs = []
    cam_reqs = []
    tunnel = []
    start_vars = getGameVars(game)
    print("Source start frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*start_vars["pos"], start_vars["hspd"], start_vars["face_angle"][1] & 65535, start_vars["num_coins"], sm64Tools.actionToStr(start_vars["action"])))

    cur_state = game.save_state()
    for frame in range(SOURCE_START_FRAME, SOURCE_END_FRAME):
        cur_input = source_inputs[frame]
        inputTools.advanceFrame(game, cur_input)
        next_state = game.save_state()
        game_vars = getGameVars(game)

        # Check for joystick / cam dependency on current frame
        cur_stick_x, cur_stick_y, cur_buttons = inputTools.splitInputs(cur_input)

        joystick_req = False
        for stick_x, stick_y in ((0, 0), (0, 127), (0, -127), (-127, 0), (127, 0)):
            game.load_state(cur_state)
            inputTools.advanceFrame(game, inputTools.joinInputs(stick_x, stick_y, cur_buttons))
            if (stateDist(game_vars, getGameVars(game)) > 0):
                joystick_req = True
                break

        # ! if 0 input, this will be False even if cam_yaw effects non-zero input
        cam_req = False
        if joystick_req:
            for cam_yaw_offset in (16384, 32768, 49152):
                game.load_state(cur_state)
                game.write("gMarioState.area->camera->yaw", game_vars["cam_yaw"] + cam_yaw_offset)
                inputTools.advanceFrame(game, cur_input)
                if (stateDist(game_vars, getGameVars(game)) > 0):
                    cam_req = True
                    break

        tunnel.append(game_vars)
        joystick_reqs.append(joystick_req)
        cam_reqs.append(cam_req)
        game.load_state(next_state)
        cur_state = next_state
    print("  Source end frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*game_vars["pos"], game_vars["hspd"], game_vars["face_angle"][1] & 65535, game_vars["num_coins"], sm64Tools.actionToStr(game_vars["action"])))

    # Setup bruteforce multiprocesses
    cur_joystick_i_shared = mp.Value(ctypes.c_uint32)
    last_sync_i_shared = mp.Value(ctypes.c_int32)
    last_dist_shared = mp.Value(ctypes.c_double)
    setup_barrier_shared = mp.Barrier(NUM_THREADS)
    setup_completed_shared = mp.Value(ctypes.c_bool, False)
    bruteforce_active_shared = mp.Value(ctypes.c_bool, False)
    sync_barrier_shared = mp.Barrier(NUM_THREADS + 1)
    block_start_shared = mp.Value(ctypes.c_uint32)
    block_end_shared = mp.Value(ctypes.c_uint32)
    new_inputs_shared = mp.Array(ctypes.c_uint32, LENGTH)
    best_inputs_shared = mp.Array(ctypes.c_uint32, LENGTH)
    best_dist_shared = mp.Value(ctypes.c_double)
    best_index_shared = mp.Value(ctypes.c_uint32)
    
    args = {
        "LENGTH": LENGTH,
        "tunnel": tunnel,
        "joystick_reqs": joystick_reqs,
        "cam_reqs": cam_reqs,
        "cur_joystick_i": cur_joystick_i_shared,
        "last_sync_i": last_sync_i_shared,
        "last_dist": last_dist_shared,
        "setup_inputs": resync_inputs[:RESYNC_START_FRAME],
        "setup_barrier": setup_barrier_shared,
        "setup_completed": setup_completed_shared,
        "bruteforce_active": bruteforce_active_shared,
        "sync_barrier": sync_barrier_shared,
        "block_start": block_start_shared,
        "block_end": block_end_shared,
        "new_inputs": new_inputs_shared,
        "best_inputs": best_inputs_shared,
        "best_dist": best_dist_shared,
        "best_index": best_index_shared
    }
    if STATE_LOADED:
        args["basis_inputs"] = basis_inputs
    g = globals()
    args.update({k: g[k] for k in g.keys() if k.isupper()})

    with ProcessGroup(NUM_THREADS, mpResyncBruteforce, (args,)) as processes:
        # Advance to resync start frame
        game.load_state(base_state)
        for frame in range(RESYNC_START_FRAME):
            inputTools.advanceFrame(game, resync_inputs[frame])

        game_vars = getGameVars(game)
        print("Resync start frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*game_vars["pos"], game_vars["hspd"], game_vars["face_angle"][1] & 65535, game_vars["num_coins"], sm64Tools.actionToStr(game_vars["action"])))

        base_dist = stateDist(start_vars, game_vars)
        if (base_dist >= MAX_RESYNC_DIST):
            raise RuntimeError(f"Starting dist too large ({base_dist}).")
        print(f"Initial dist: %.10f" % base_dist)

        # Advance to first frame where joystick matters
        i = 0
        while (i < LENGTH):
            frame = i + RESYNC_START_FRAME
            joystick_req = joystick_reqs[i]
            if joystick_req:
                break
            inputTools.advanceFrame(game, new_inputs[i])
            dist = stateDist(tunnel[i], getGameVars(game))
            if (dist > MAX_RESYNC_DIST):
                raise RuntimeError(f"Dist drift before resync began (f={frame}, dist={dist})")
            i += 1

        resync_begin_i = i
        resync_begin_state = game.save_state()

        # Main resync loop
        last_sync_i = -1
        last_dist = base_dist
        while (i < LENGTH):
            frame = i + RESYNC_START_FRAME
            
            """Segment is current frame to next frame with joystick input or end frame
               Idea is to benchmark segment accuracy as far into the future as possible
               Usually this will just be 1 frame long"""
            seg_end = i + 1
            while ((seg_end < LENGTH) and (not joystick_reqs[seg_end])):
                seg_end += 1
            seg_length = seg_end - i
            frame_seg_end = frame + seg_length

            source_stick_x, source_stick_y, source_buttons = inputTools.splitInputs(source_inputs[SOURCE_START_FRAME + i])
            new_buttons = inputTools.splitInputs(new_inputs[i])[2]

            cur_state = game.save_state()
            cur_start_vars = getGameVars(game)
            target_vars = tunnel[seg_end - 1]

            # Get close joysticks on current frame
            cam_req = cam_reqs[i]
            if cam_req:
                joystick_ref = tunnel[i]
                target_mag = joystick_ref["stick_mag"]
                target_yaw = joystick_ref["intended_yaw"] - cur_start_vars["cam_yaw"]
                best_joysticks = inputTools.getNearestJoysticks(target_mag, target_yaw, n=JOYSTICK_SEARCH_RANGE)
            else:
                best_joysticks = inputTools.getNearestJoysticksRaw((source_stick_x, source_stick_y), n=JOYSTICK_SEARCH_RANGE)

            # Test close joysticks up to seg end
            best_dist = float("inf")
            first = True
            for stick in best_joysticks:
                game.load_state(cur_state)
                inputTools.advanceFrame(game, inputTools.joinInputs(*stick, new_buttons))
                for j in range(i + 1, seg_end):
                    inputTools.advanceFrame(game, new_inputs[j])

                dist = stateDist(target_vars, getGameVars(game))
                if (dist < best_dist):
                    best_dist = dist
                    best_stick = stick
                    if (dist < INSTANT_ACCEPT_DIST):
                        break

                # Save first for bruteforce, as if sync fails then current dist measure may be unreliable
                if first:
                    first_dist = dist
                first = False

            if (best_dist < BRUTEFORCE_START_DIST):
                new_inputs[i] = inputTools.joinInputs(*best_stick, new_buttons)
                
                # Advance to segment end
                game.load_state(cur_state)
                while (i < seg_end):
                    inputTools.advanceFrame(game, new_inputs[i])
                    i += 1
            else:
                # Use closest true joystick if dist is very large, as dist measure may be unreliable
                if (best_dist >= OVEREXTENDED_DIST):
                    best_stick = best_joysticks[0]
                    best_dist = first_dist
                new_inputs[i] = inputTools.joinInputs(*best_stick, new_buttons)

                # If resync is off, bruteforce preceeding inputs to reduce dist
                print(f"%6d - Dist too large (%.10f), optimizing..." % (frame_seg_end, best_dist))

                if (last_sync_i == -1):
                    block_start = i
                else:
                    pert_frames_count = 1
                    block_start = i - 1
                    while ((pert_frames_count < LOOKBACK_FRAMES) and (block_start > resync_begin_i)):
                        if joystick_reqs[i]:
                            pert_frames_count += 1
                        block_start -= 1
                block_end = seg_end

                # Setup info for processes
                cur_joystick_i_shared.value = i
                last_sync_i_shared.value = last_sync_i
                last_dist_shared.value = last_dist
                block_start_shared.value = block_start
                block_end_shared.value = block_end
                new_inputs_shared[:seg_end] = new_inputs[:seg_end]
                best_inputs_shared[:seg_end] = new_inputs[:seg_end]
                if ((last_sync_i == -1) or (best_dist < OVEREXTENDED_DIST)):
                    best_dist_shared.value = best_dist
                else:
                    best_dist_shared.value = OVEREXTENDED_DIST + last_dist
                best_index_shared.value = 0

                # Wait for setup to complete if needed
                while not setup_completed_shared.value:
                    time.sleep(0.01)

                # Activate bruteforcing
                bruteforce_active_shared.value = True
                barrierSync(sync_barrier_shared)

                # Bruteforce host loop
                bruteforce_active_shared.value = True
                try:
                    while (best_dist_shared.value >= BRUTEFORCE_ACCEPT_DIST):
                        processes.print()
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    if (best_dist_shared.value >= MAX_RESYNC_DIST):
                        print("resync stopped")
                        break

                # Stop bruteforce workers
                bruteforce_active_shared.value = False
                barrierSync(sync_barrier_shared)
                processes.print()

                # Copy result
                best_dist = best_dist_shared.value
                new_inputs[block_start:block_end] = best_inputs_shared[block_start:block_end]

                # Resync
                game.load_state(resync_begin_state)
                for j in range(resync_begin_i, block_end):
                    inputTools.advanceFrame(game, new_inputs[j])
                i += seg_length
                
            print(f"%6d - Dist: %.10f" % (frame_seg_end, best_dist))
            last_sync_i = i - 1
            last_dist = best_dist

    # Output
    inputs_out = np.concatenate((resync_inputs[:RESYNC_START_FRAME], new_inputs))
    print(f"Saving m64 to \"{OUT_NAME}\"")
    inputTools.saveM64(OUT_NAME, inputs_out, resync_header)
    

if (__name__ == "__main__"):
    # Load m64s
    print(f"Loading source m64 \"{SOURCE_NAME}\"")
    source_inputs, source_header = inputTools.loadM64(SOURCE_NAME)
    print(f"Loading resync m64 \"{RESYNC_NAME}\"")
    resync_inputs, resync_header = inputTools.loadM64(RESYNC_NAME)
    
    STATE_LOADED = source_header.state_loaded
    if STATE_LOADED:
        print(f"Loading basis m64 \"{BASIS_NAME}\"")
        basis_inputs = inputTools.loadM64(BASIS_NAME)[0]

    resync()
        
