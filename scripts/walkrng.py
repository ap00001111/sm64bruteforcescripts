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
from mpManager import ProcessGroup, printMP


INF = float("inf")

NUM_THREADS = 1

# WAFEL info
DLL_PATH = ""
GAME_VERSION = "us"

# Target m64 info
START_FRAME = 0
END_FRAME = 0
INP_NAME = ".m64"
OUT_NAME_FMT = ".%d.m64"

# Basis m64 info (for state loaded)
STATE_OFFSET = 0
BASIS_NAME = "basis.m64"

# Num joysticks to test
JOYSTICK_SEARCH_RANGE = 256


# Run after basis m64 inputs, for fixing rng/global timer etc to match state loaded m64
def syncStateLoaded(game):
    sm64Tools.setRngIndex(game, 24281)
    sm64Tools.setGlobalTimer(game, 92006)

# Get game vars relevant to tunnel
def getGameVars(game):
    return {"pos": game.read("gMarioState.pos"),
            "vel": game.read("gMarioState.vel"),
            "hspd": game.read("gMarioState.forwardVel"),
            "face_angle": game.read("gMarioState.faceAngle"),
            "angle_vel": game.read("gMarioState.angleVel"),
            "num_coins": game.read("gMarioState.numCoins"),
            "action": game.read("gMarioState.action")}

# Not part of game vars so different sticks leading to same state can be easily compared
def getStickVars(game):
    return {"stick_mag": game.read("gMarioState.controller->stickMag"),
            "intended_yaw": game.read("gMarioState.intendedYaw")}


def getJoystickVariants():
    LENGTH = END_FRAME - START_FRAME

     # Load WAFEL
    lib_full_path = os.path.join(DLL_PATH, f"sm64_{GAME_VERSION}.dll")
    game = wafel.Game(lib_full_path)

    # Advance to start frame
    if STATE_LOADED:
        for frame in range(STATE_OFFSET):
            inputTools.advanceFrame(game, basis_inputs[frame])
        syncStateLoaded(game)
    for frame in range(START_FRAME):
        inputTools.advanceFrame(game, inputs[frame])
    start_state = game.save_state()

    # Create tunnel
    game_vars = getGameVars(game)
    print("Start frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*game_vars["pos"], game_vars["hspd"], game_vars["face_angle"][1] & 65535, game_vars["num_coins"], sm64Tools.actionToStr(game_vars["action"])))
    
    tunnel = []
    tunnel_stick_data = []
    tunnel_rng_indices = []
    for frame in range(START_FRAME, END_FRAME):
        inputTools.advanceFrame(game, inputs[frame])
        tunnel.append(getGameVars(game))
        tunnel_stick_data.append(getStickVars(game))
        tunnel_rng_indices.append(sm64Tools.getRngIndex(game))

    game_vars = getGameVars(game)
    print("  End frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*game_vars["pos"], game_vars["hspd"], game_vars["face_angle"][1] & 65535, game_vars["num_coins"], sm64Tools.actionToStr(game_vars["action"])))

    # Find joystick manips
    stick_choices = []
    game.load_state(start_state)
    for i in range(LENGTH):
        frame = i + START_FRAME
        cur_inputs = inputs[frame]
        cur_tunnel_point = tunnel[i]
        cur_tunnel_stick_data = tunnel_stick_data[i]
        if ((cur_tunnel_point["action"] != 0x04000440) or (cur_tunnel_point["hspd"] > 16)):
            inputTools.advanceFrame(game, cur_inputs)
            continue
        
        cur_state = game.save_state()
        cur_stick_x, cur_stick_y, cur_buttons = inputTools.splitInputs(cur_inputs)

        cam_yaw = sm64Tools.getCamYaw(game)
        target_mag = cur_tunnel_stick_data["stick_mag"]
        target_yaw = cur_tunnel_stick_data["intended_yaw"] - cam_yaw
        best_joysticks = inputTools.getNearestJoysticks(target_mag, target_yaw, n=JOYSTICK_SEARCH_RANGE, yaw_weight=100)

        checked_next_vars = []
        for stick in best_joysticks:
            game.load_state(cur_state)
            inputTools.advanceFrame(game, inputTools.joinInputs(*stick, cur_buttons))
            if (sm64Tools.getRngIndex(game) == tunnel_rng_indices[i]):
                continue

            next_vars = getGameVars(game)
            if next_vars in checked_next_vars:
                continue
            checked_next_vars.append(next_vars)
            next_stick_data = getStickVars(game)

            j = i + 1
            success = (next_vars == tunnel[i])
            while (not success and (j < LENGTH)):
                search_frame = START_FRAME + j
                inputTools.advanceFrame(game, inputs[search_frame])
                
                search_vars = getGameVars(game)
                search_tunnel_point = tunnel[j]
                if (search_vars == search_tunnel_point):
                    success = True
                    break

                dist = np.float32(search_vars["pos"]) - np.float32(search_tunnel_point["pos"])
                if (np.linalg.norm(dist) > 1e-5):
                    break
                
                j += 1

            if not success:
                continue

            lower_stick = np.int64((cur_stick_x, cur_stick_y)).astype(np.int8)
            upper_stick = stick
            if (next_stick_data["stick_mag"] < target_mag):
                lower_stick, upper_stick = upper_stick, lower_stick
            stick_choice = (frame, lower_stick, upper_stick)
            print("Frame %6d - (% 4d,% 4d) / (% 4d,% 4d)" % (frame, *lower_stick, *upper_stick))
            stick_choices.append(stick_choice)
            break
                    
        game.load_state(cur_state)
        inputTools.advanceFrame(game, cur_inputs)

    num_choices_found = len(stick_choices)
    print(f"Num manip frames found: {num_choices_found}")
    return stick_choices


def mpTestJoystickVariants(thread_id, args):
    globals().update(args)

    # Load WAFEL
    lib_full_path = os.path.join(DLL_PATH, f"sm64_{GAME_VERSION}.dll")
    game = wafel.Game(lib_full_path)

    # Advance to start frame
    if STATE_LOADED:
        for frame in range(STATE_OFFSET):
            inputTools.advanceFrame(game, basis_inputs[frame])
    syncStateLoaded(game)
    for frame in range(START_FRAME):
        inputTools.advanceFrame(game, inputs[frame])
    start_state = game.save_state()

    cur_inputs = inputs[START_FRAME:END_FRAME].copy()

    # Print start/end state to verify sim accuracy
    with setup_complete.get_lock():
        if not setup_complete.value:
            game_vars = getGameVars(game)
            printMP("Start frame state - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*game_vars["pos"], game_vars["hspd"], game_vars["face_angle"][1] & 65535, game_vars["num_coins"], sm64Tools.actionToStr(game_vars["action"])))
            
            for i in range(LENGTH):
                frame = i + START_FRAME
                inputTools.advanceFrame(game, cur_inputs[i])

            game_vars = getGameVars(game)
            printMP("  End frame state - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*game_vars["pos"], game_vars["hspd"], game_vars["face_angle"][1] & 65535, game_vars["num_coins"], sm64Tools.actionToStr(game_vars["action"])))
            
            setup_complete.value = True

    # Test
    local_saved_filenames = []
    while True:
        with shared_index.get_lock():
            if (shared_index.value == num_cases):
                break
            cur_index = shared_index.value
            shared_index.value += 1

        # Build stick choices
        v = cur_index
        for frame, lower_choice, upper_choice in indiv_choices:
            i = frame - START_FRAME
            cur_buttons = inputTools.splitInputs(cur_inputs[i])[2]
            if (v & 1):
                cur_choice = lower_choice
            else:
                cur_choice = upper_choice
            cur_inputs[i] = inputTools.joinInputs(*cur_choice, cur_buttons)
            v >>= 1
            
        for group in grouped_choices:
            group_size = len(group)
            num_group_options = group_size + 1
            first_upper_i = v % num_group_options
            for i in range(group_size):
                frame, lower_choice, upper_choice = group[i]
                j = frame - START_FRAME
                cur_buttons = inputTools.splitInputs(cur_inputs[j])[2]
                if (i < first_upper_i):
                    cur_choice = lower_choice
                else:
                    cur_choice = upper_choice
                cur_inputs[j] = inputTools.joinInputs(*cur_choice, cur_buttons)
            v //= num_group_options

        # Test new inputs
        game.load_state(start_state)
        for i in range(LENGTH):
            frame = i + START_FRAME
            inputTools.advanceFrame(game, cur_inputs[i])

        # Save result
        rng_index = sm64Tools.getRngIndex(game)
        out_name = OUT_NAME_FMT % rng_index
        if out_name not in local_saved_filenames:
            local_saved_filenames.append(out_name)
            with save_lock:
                if not os.path.isfile(out_name):
                    printMP(f"Saving m64 \"{out_name}\"")
                    inputs_out = np.concatenate((inputs[:START_FRAME], cur_inputs, inputs[END_FRAME:]))
                    inputTools.saveM64(out_name, inputs_out, header)


def testJoystickVariants(indiv_choices, grouped_choices):
    num_cases = int((1 << len(indiv_choices)) * np.prod([(len(group) + 1) for group in grouped_choices]))
    print(f"Total test cases: {num_cases}")

    setup_complete = mp.Value(ctypes.c_bool, False)
    shared_index = mp.Value(ctypes.c_uint64, 0)
    save_lock = mp.Lock()

    args = {
        "LENGTH": END_FRAME - START_FRAME,
        "header": header,
        "inputs": inputs,
        "setup_complete": setup_complete,
        "indiv_choices": indiv_choices,
        "grouped_choices": grouped_choices,
        "shared_index": shared_index,
        "save_lock": save_lock,
        "num_cases": num_cases
    }
    if STATE_LOADED:
        args["basis_inputs"] = basis_inputs
    g = globals()
    args.update({k: g[k] for k in g.keys() if k.isupper()})

    with ProcessGroup(NUM_THREADS, mpTestJoystickVariants, (args,)) as processes:
        try:
            while processes.active:
                processes.print()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass


# Print formatted choices for setting up choice groups
def printChoicesList(choices):
    for frame, lower_stick, upper_stick in choices:
        print("(%6d, (% 4d,% 4d), (% 4d,% 4d))," % (frame, *lower_stick, *upper_stick))


if (__name__ == "__main__"):
    # Load m64s
    print(f"Loading m64 \"{INP_NAME}\"")
    inputs, header = inputTools.loadM64(INP_NAME)

    STATE_LOADED = header.state_loaded
    if STATE_LOADED:
        print(f"Loading basis m64 \"{BASIS_NAME}\"")
        basis_inputs = inputTools.loadM64(BASIS_NAME)[0]

    choices = getJoystickVariants()
    printChoicesList(choices)
