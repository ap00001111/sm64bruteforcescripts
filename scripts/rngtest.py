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

# Thread count (should be slightly less than total CPU threads)
NUM_THREADS = 1

# WAFEL info
DLL_PATH = ""
GAME_VERSION = "us"

# Target m64 info
START_FRAME = 0 # Frame RNG is set
END_FRAME = 0
INP_NAME = ".m64"

# Basis m64 info (for state loaded)
STATE_OFFSET = 0
BASIS_NAME = "basis.m64"

# RNG options
INDEX_MIN = 0
INDEX_MAX = 65113 # Searches min<=i<=max
BLACKLISTED_INDICES = None # Replace with array to use
WHITELISTED_INDICES = None

PRINT_AS_FOUND = False

# Bounds (for checking success)
POS_BOUNDS = ((-INF, INF), (-INF, INF), (-INF, INF))
VEL_BOUNDS = ((-INF, INF), (-INF, INF), (-INF, INF))
HSPD_BOUNDS = (-INF, INF)
FACE_ANGLE_BOUNDS = ((-32768, 32767), (0, 65535), (-32768, 32767))
ANGLE_VEL_BOUNDS = ((-32768, 32767), (-32768, 32767), (-32768, 32767))
COINS_BOUNDS = (0, 999)
TARGET_ACTION = None


valInBounds = lambda val, bounds: bounds[0] <= val <= bounds[1]
shortInBounds = lambda val, bounds: (int(val - bounds[0]) & 65535) <= (int(bounds[1] - bounds[0]) & 65535)
def valsInBoundIter(vals, bounds, checkFunc):
    for i in range(len(vals)):
        if not checkFunc(vals[i], bounds[i]):
            return False
    return True

# Check if current game state completed target objective
def isSuccess(game):
    return (valsInBoundIter(pos, POS_BOUNDS, valInBounds) and
            valsInBoundIter(vel, VEL_BOUNDS, valInBounds) and
            valInBounds(hspd, HSPD_BOUNDS) and
            valsInBoundIter(face_angle, FACE_ANGLE_BOUNDS, shortInBounds) and
            valsInBoundIter(angle_vel, ANGLE_VEL_BOUNDS, shortInBounds) and
            valInBounds(num_coins, COINS_BOUNDS) and
            ((TARGET_ACTION == None) or (TARGET_ACTION == action)))
    
    
# Read game vars to global vars
def refreshGameVars(game):
    game_vars = {"pos": game.read("gMarioState.pos"),
                 "vel": game.read("gMarioState.vel"),
                 "hspd": game.read("gMarioState.forwardVel"),
                 "face_angle": game.read("gMarioState.faceAngle"),
                 "angle_vel": game.read("gMarioState.angleVel"),
                 "num_coins": game.read("gMarioState.numCoins"),
                 "action": game.read("gMarioState.action")}
    globals().update(game_vars)


# Run after basis m64 inputs, for fixing rng/global timer etc to match state loaded m64
def syncStateLoaded(game):
    pass


# Testing function
def mpTestRng(thread_id, args):
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

        game.load_state(start_state)
        sm64Tools.setRngIndex(game, rng_index)
        for i in range(LENGTH):
            frame = i + START_FRAME
            inputTools.advanceFrame(game, cur_inputs[i])

        refreshGameVars(game)
        if isSuccess(game):
            working_index_mask[cur_index] = True
            if PRINT_AS_FOUND:
                printMP(rng_index)
        

# Start or restart rng testing
def testRng():
    global working_indices

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
    working_index_mask = mp.Array(ctypes.c_bool, test_indices.size)
    for i in range(test_indices.size):
        working_index_mask[i] = False

    args = {
        "LENGTH": END_FRAME - START_FRAME,
        "inputs": inputs,
        "setup_complete": setup_complete,
        "test_indices": test_indices,
        "shared_index": shared_index,
        "working_index_mask": working_index_mask
    }
    if STATE_LOADED:
        args["basis_inputs"] = basis_inputs
    g = globals()
    args.update({k: g[k] for k in g.keys() if k.isupper()})

    with ProcessGroup(NUM_THREADS, mpTestRng, (args,)) as processes:
        try:
            while processes.active:
                processes.print()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

    working_indices = test_indices[working_index_mask]
    num_working_indices = working_indices.size
    print(f"Found indices ({num_working_indices}): {working_indices}")

    
if (__name__ == "__main__"):
    # Load m64s
    print(f"Loading m64 \"{INP_NAME}\"")
    inputs, header = inputTools.loadM64(INP_NAME)

    STATE_LOADED = header.state_loaded
    if STATE_LOADED:
        print(f"Loading basis m64 \"{BASIS_NAME}\"")
        basis_inputs = inputTools.loadM64(BASIS_NAME)[0]

    testRng()
