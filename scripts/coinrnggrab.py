import ctypes
import gzip
import multiprocessing as mp
import numpy as np
import os
import pickle as pkl
import sys
import time
import wafel

sys.path.append("core")
import inputTools
import sm64Tools
from mpManager import ProcessGroup, printMP, enumQueue


INF = float("inf")

# Thread count (should be slightly less than total CPU threads)
NUM_THREADS = 6

# WAFEL info
DLL_PATH = "libsm64"
GAME_VERSION = "us"

# Target m64 info
START_FRAME = 0 # Frame RNG is set
END_FRAME = 0
INP_NAME = ".m64"

# Coin data output file
COIN_DATA_NAME = ".pkl.gz"

# Basis m64 info (for state loaded)
STATE_OFFSET = 0
BASIS_NAME = "basis.m64"

# RNG options
INDEX_MIN = 0
INDEX_MAX = 65113 # Searches min<=i<=max
BLACKLISTED_INDICES = None # Replace with array to use
WHITELISTED_INDICES = None

# Coin tracking bounds
MARIO_COIN_DIST_BOUNDS = (0, INF)
COIN_FRAME_BOUNDS = (0, INF)
COIN_POS_BOUNDS = ((-INF, INF), (-INF, INF), (-INF, INF))
COIN_VALUE = None


valInBounds = lambda val, bounds: bounds[0] <= val <= bounds[1]
shortInBounds = lambda val, bounds: (int(val - bounds[0]) & 65535) <= (int(bounds[1] - bounds[0]) & 65535)
def valsInBoundIter(vals, bounds, checkFunc):
    for i in range(len(vals)):
        if not checkFunc(vals[i], bounds[i]):
            return False
    return True

# If true, stores coin data for current frame
def shouldTrackCoin(game, frame, coin_slot, coin_pos, coin_radius, coin_value):
    mario_pos = game.read("gMarioState.pos")
    mario_coin_ydist = max(0, coin_pos[1] - mario_pos[1] - 160, mario_pos[1] - coin_pos[1] - 64)
    mario_coin_hdist_sq = max(0, np.square(mario_pos[0] - coin_pos[0]) + np.square(mario_pos[2] - coin_pos[2]) - np.square(coin_radius + 37))
    mario_coin_dist = np.sqrt(np.square(mario_coin_ydist) + mario_coin_hdist_sq)
    
    return (valInBounds(mario_coin_dist, MARIO_COIN_DIST_BOUNDS) and
            valInBounds(frame, COIN_FRAME_BOUNDS) and
            valsInBoundIter(coin_pos, COIN_POS_BOUNDS, valInBounds) and
            ((COIN_VALUE == None) or (COIN_VALUE == coin_value)))
    
    
# Read game vars to global vars
def refreshGameVars(game):
    game_vars = {"pos": game.read("gMarioState.pos"),
                 "vel": game.read("gMarioState.vel"),
                 "hspd": game.read("gMarioState.forwardVel"),
                 "face_angle": game.read("gMarioState.faceAngle"),
                 "angle_vel": game.read("gMarioState.angleVel"),
                 "num_coins": game.read("gMarioState.numCoins"),
                 "action": game.read("gMarioState.action"),
                 "mario_obj_timer": sm64Tools.getMarioObjTimer}
    globals().update(game_vars)


# Run after basis m64 inputs, for fixing rng/global timer etc to match state loaded m64
def syncStateLoaded(game):
    pass

# Get coin info
def updateCoinData(game, frame, coin_data):
    if not valInBounds(frame, COIN_FRAME_BOUNDS):
        return
    
    found_uids = []
    base_uid = frame << 8
    for slot in sm64Tools.enumSlots(game, processing_groups=(6,)):
        if not sm64Tools.objIsCoin(game, slot):
            continue

        # Give each coin a unique id for tracking
        cur_coin_uid = game.read(f"gObjectPool[{slot}].oFaceAngleRoll")
        if (cur_coin_uid == 0):
            base_uid += 1
            cur_coin_uid = base_uid
            game.write(f"gObjectPool[{slot}].oFaceAngleRoll", cur_coin_uid)

        # Hacking the coin back to being intangible can make it stay intangible
        # Once a coin has been tangible once, ignore tangibility check
        existing_coin = (cur_coin_uid in coin_data["uids"])
        if not (existing_coin or sm64Tools.getCoinTangibility(game, slot)):
            continue

        cur_coin_pos = sm64Tools.getObjPos(game, slot)
        cur_coin_radius = sm64Tools.getCoinRadius(game, slot)
        cur_coin_value = sm64Tools.getCoinValue(game, slot)
        if not shouldTrackCoin(game, frame, slot, cur_coin_pos, cur_coin_radius, cur_coin_value):
            continue
        
        if existing_coin:
            cur_coin_data = coin_data[cur_coin_uid]
        else:
            cur_coin_data = {"value": cur_coin_value,
                             "effectiveRadius": cur_coin_radius + 37,
                             "frames": [],
                             "positions": [],
                             "preceeding_uids": []}
            coin_data[cur_coin_uid] = cur_coin_data
            coin_data["uids"].append(cur_coin_uid)

        cur_coin_data["frames"].append(frame)
        cur_coin_data["positions"].append(cur_coin_pos)
        for uid in found_uids:
            if uid not in cur_coin_data["preceeding_uids"]:
                cur_coin_data["preceeding_uids"].append(uid)
        found_uids.append(cur_coin_uid)

        # Ensure coin isn't collected so it can be tracked
        sm64Tools.setCoinTangibility(game, slot, False)


# Get coin info for rng cases
def mpGetCoinData(thread_id, args):
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
            refreshGameVars(game)
            printMP("Start frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*pos, hspd, face_angle[1] & 65535, num_coins, sm64Tools.actionToStr(action)))
            
            for i in range(LENGTH):
                frame = i + START_FRAME
                inputTools.advanceFrame(game, cur_inputs[i])

            refreshGameVars(game)
            printMP("  End frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*pos, hspd, face_angle[1] & 65535, num_coins, sm64Tools.actionToStr(action)))
            
            setup_complete.value = True

    # Test
    while True:
        with shared_index.get_lock():
            if (shared_index.value == test_indices.size):
                break
            cur_index = shared_index.value
            rng_index = test_indices[cur_index]
            shared_index.value += 1

        # Get coin data
        coin_data = {"rng_index": np.uint16(rng_index), "uids": []}
        game.load_state(start_state)
        sm64Tools.setRngIndex(game, rng_index)
        for i in range(LENGTH):
            frame = i + START_FRAME
            updateCoinData(game, frame, coin_data)
            inputTools.advanceFrame(game, cur_inputs[i])
        updateCoinData(game, END_FRAME, coin_data)

        # Topologically sort coins to get processing order
        sorted_uids = []
        rel_data = [(uid, coin_data[uid]["preceeding_uids"]) for uid in coin_data["uids"]]
        while rel_data:
            l = len(rel_data)
            for i in range(l):
                cur_rel = rel_data[i]
                if not cur_rel[1]:
                    uid = cur_rel[0]
                    sorted_uids.append(uid)
                    for j in range(l):
                        if (i == j):
                            continue
                        comp_precs = rel_data[j][1]
                        if uid in comp_precs:
                            comp_precs.remove(uid)
                    rel_data = rel_data[:i] + rel_data[i+1:]
                    break
                if (i == (l - 1)):
                    raise RuntimeError("Cycle in coin processing orders")
        coin_data["uids"] = sorted_uids
        for uid in sorted_uids:
            del coin_data[uid]["preceeding_uids"]
        
        coin_data_queue.put(coin_data)

        # Print unique coin counts (for verifying sim accuracy)
        num_coins_found = len(sorted_uids)
        with found_coin_counts.get_lock():
            if not found_coin_counts[num_coins_found]:
                if (num_coins_found == 1):
                    printMP(f"Found 1 tangible coin (rng index {rng_index})")
                else:
                    printMP(f"Found {num_coins_found} tangible coins (rng index {rng_index})")
                found_coin_counts[num_coins_found] = True
        

# Start or restart rng testing
def getCoinData():
    global out_coin_data
    
    rel_index_max = (INDEX_MAX - INDEX_MIN) % 65114
    if (WHITELISTED_INDICES == None):
        test_indices = [((i + INDEX_MIN) % 65114) for i in range(rel_index_max + 1)]
    else:
        test_indices = filter(lambda i: (((i - INDEX_MIN) % 65114) <= rel_index_max), WHITELISTED_INDICES)
    if (BLACKLISTED_INDICES != None):
        test_indices = filter(lambda i: i not in BLACKLISTED_INDICES, test_indices)
    test_indices = np.uint16(test_indices)

    setup_complete = mp.Value(ctypes.c_bool, False)
    shared_index = mp.Value(ctypes.c_uint16, 0)
    coin_data_queue = mp.Queue(maxsize=test_indices.size)
    found_coin_counts = mp.Array(ctypes.c_bool, 241)
    for i in range(241):
        found_coin_counts[i] = False

    args = {
        "LENGTH": END_FRAME - START_FRAME,
        "inputs": inputs,
        "setup_complete": setup_complete,
        "test_indices": test_indices,
        "shared_index": shared_index,
        "coin_data_queue": coin_data_queue,
        "found_coin_counts": found_coin_counts
    }
    if STATE_LOADED:
        args["basis_inputs"] = basis_inputs
    g = globals()
    args.update({k: g[k] for k in g.keys() if k.isupper()})

    all_coin_data = []
    with ProcessGroup(NUM_THREADS, mpGetCoinData, (args,)) as processes:
        try:
            while processes.active:
                processes.print()
                for coin_data in enumQueue(coin_data_queue): # Processes won't close until queue is empty
                    all_coin_data.append(coin_data)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

    for coin_data in enumQueue(coin_data_queue):
        all_coin_data.append(coin_data)
        
    num_checked_indices = len(all_coin_data)
    if (num_checked_indices != test_indices.size):
        while True:
            res = input(f"Did not test all indices ({num_checked_indices}/{test_indices.size}). Save (Y/N)? ").upper()
            if res.startswith("Y"):
                break
            elif res.startswith("N"):
                return

    # Normalize output
    print("Processing coin data")
    all_coin_data = sorted(all_coin_data, key = lambda coin_data: coin_data["rng_index"])
    
    max_coins = 0
    data_frames = []
    for coin_data in all_coin_data:
        max_coins = max(max_coins, len(coin_data["uids"]))
        for uid in coin_data["uids"]:
            for frame in coin_data[uid]["frames"]:
                if frame not in data_frames:
                    data_frames.append(frame)
    data_frames = sorted(data_frames)
    frame_indices = {frame: data_frames.index(frame) for frame in data_frames}
    data_frames = np.int32(data_frames)

    checked_indices = np.uint16([coin_data["rng_index"] for coin_data in all_coin_data])
    coin_effective_radii = np.zeros((num_checked_indices, max_coins), dtype=np.float32)
    coin_values = np.zeros((num_checked_indices, max_coins), dtype=np.uint8)
    coin_positions = np.ndarray((num_checked_indices, max_coins, len(data_frames), 3), dtype=np.float32)
    coin_positions[:] = INF
    for i in range(num_checked_indices):
        coin_data = all_coin_data[i]
        cur_uids = coin_data["uids"]
        cur_num_coins = len(cur_uids)
        for j in range(cur_num_coins):
            cur_uid = cur_uids[j]
            cur_coin_data = coin_data[cur_uid]
            coin_effective_radii[i,j] = cur_coin_data["effectiveRadius"]
            coin_values[i,j] = cur_coin_data["value"]
            
            cur_frames = cur_coin_data["frames"]
            cur_positions = cur_coin_data["positions"]
            for k in range(len(cur_frames)):
                frame = cur_frames[k]
                frame_index = frame_indices[frame]
                coin_positions[i,j,frame_index] = cur_positions[k]

    print(f"Saving coin data to \"{COIN_DATA_NAME}\"")
    out_coin_data = {
        "frames": data_frames,
        "positions": coin_positions,
        "effective_radii": coin_effective_radii,
        "values": coin_values,
        "rng_indices": checked_indices
    }
    with gzip.open(COIN_DATA_NAME, "wb") as fp:
        pkl.dump(out_coin_data, fp, protocol=pkl.HIGHEST_PROTOCOL)

    
if (__name__ == "__main__"):
    # Load m64s
    print(f"Loading m64 \"{INP_NAME}\"")
    inputs, header = inputTools.loadM64(INP_NAME)

    STATE_LOADED = header.state_loaded
    if STATE_LOADED:
        print(f"Loading basis m64 \"{BASIS_NAME}\"")
        basis_inputs = inputTools.loadM64(BASIS_NAME)[0]

    getCoinData()
