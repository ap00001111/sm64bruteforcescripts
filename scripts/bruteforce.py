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
FAIL_SCORE = -INF

# Thread count (should be slightly less than total CPU threads)
NUM_THREADS = 1

# WAFEL info
DLL_PATH = ""
GAME_VERSION = "us"

# Target m64 info
START_FRAME = 0
END_FRAME = 0
INP_NAME = ".m64"
OUT_NAME = ".m64"
SAVE_END_INPUTS = False

# Basis m64 info (for state loaded)
STATE_OFFSET = 0
BASIS_NAME = "basis.m64"

# Input perturbing options
PERT_PROB = 0.1
PERT_SIZE = 16

# Simulated annealing options
SCORE_LENIENCY = 2 # Max amount score can be worse than best and be used
INITIAL_TEMP = 1
TEMP_DECAY_RATE = 0.999

# Weights (for maximizing)
POS_WEIGHTS = (0, 0, 0)
VEL_WEIGHTS = (0, 0, 0)
HSPD_WEIGHT = 0
FACE_ANGLE_WEIGHTS = (0, 0, 0)
ANGLE_VEL_WEIGHTS = (0, 0, 0)
COINS_WEIGHT = 0

# Targets
POS_TARGETS = (None, None, None)
VEL_TARGETS = (None, None, None)
HSPD_TARGET = None
FACE_ANGLE_TARGETS = (None, None, None)
ANGLE_VEL_TARGETS = (None, None, None)
COINS_TARGET = None

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


def getWeightTargetScore(val, weight, target):
    if (weight == 0):
        return 0
    if (target == None):
        return weight*val
    dist = val - target
    return -weight*dist*dist

def getTargetScoreShort(val, weight, target):
    if (weight == 0):
        return 0
    dist = ((val - target + 32768) & 65535) - 32768
    return -weight*dist*dist

def sumScoresIter(vals, weights, targets, scoreFunc):
    score = 0
    for i in range(len(vals)):
        score += scoreFunc(vals[i], weights[i], targets[i])
    return score

# Score current game state
def scoreState(game):
    if not isSuccess(game):
        return FAIL_SCORE
    score = sumScoresIter(pos, POS_WEIGHTS, POS_TARGETS, getWeightTargetScore)
    score += sumScoresIter(vel, VEL_WEIGHTS, VEL_TARGETS, getWeightTargetScore)
    score += getWeightTargetScore(hspd, HSPD_WEIGHT, HSPD_TARGET)
    score += sumScoresIter(face_angle, FACE_ANGLE_WEIGHTS, FACE_ANGLE_TARGETS, getTargetScoreShort)
    score += sumScoresIter(angle_vel, ANGLE_VEL_WEIGHTS, ANGLE_VEL_TARGETS, getTargetScoreShort)
    score += getWeightTargetScore(num_coins, COINS_WEIGHT, COINS_TARGET)
    return score
    
    
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

# Random input perturbation
def perturbInput(inp, frame, p=PERT_PROB, s=PERT_SIZE):
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

# Run after basis m64 inputs, for fixing rng/global timer etc to match state loaded m64
def syncStateLoaded(game):
    pass


# Bruteforcing function
def mpBruteforce(thread_id, args):
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
        inputTools.advanceFrame(game, setup_inputs[frame])
    start_state = game.save_state()

    cur_inputs = setup_inputs[START_FRAME:END_FRAME].copy()
    test_inputs = cur_inputs.copy()
    local_best_index = start_best_index

    # Benchmark
    with setup_complete.get_lock():
        if not setup_complete.value:
            refreshGameVars(game)
            printMP("Start frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*pos, hspd, face_angle[1] & 65535, num_coins, sm64Tools.actionToStr(action)))
            
            for i in range(LENGTH):
                frame = i + START_FRAME
                inputTools.advanceFrame(game, cur_inputs[i])
                
            refreshGameVars(game)
            printMP("  End frame - Pos: (% 9.3f,% 9.3f,% 9.3f)  HSpd:% 9.3f  Yaw: %5hu  Coins: %3d  Act: %s" % (*pos, hspd, face_angle[1] & 65535, num_coins, sm64Tools.actionToStr(action)))
            
            best_score.value = scoreState(game)
            info = getPrintInfo()
            if (best_score.value == FAIL_SCORE):
                printMP(f"Initial inputs fail! {{{info}}}")
            else:
                printMP(f"Initial score: %.5f {{{info}}}" % best_score.value)
                
            setup_complete.value = True
            time_start.value = time.time()

    # Bruteforce
    temp = INITIAL_TEMP
    cur_score = best_score.value
    while True:
        cur_best_index = best_index.value
        if (cur_best_index != local_best_index):
            local_best_index = cur_best_index
            with best_inputs.get_lock():
                cur_inputs[:] = best_inputs
            cur_score = best_score.value

        cycle_frames = 0
        game.load_state(start_state)
        for i in range(LENGTH):
            frame = i + START_FRAME
            new_input = perturbInput(cur_inputs[i], frame)
            test_inputs[i] = new_input
            inputTools.advanceFrame(game, new_input)
            cycle_frames += 1

        refreshGameVars(game)
        score = scoreState(game)
        with best_score.get_lock():
            if (score > best_score.value):
                # Output new best
                best_score.value = score
                local_best_index = best_index.value + 1
                info = getPrintInfo()
                printMP(f"New best: %.5f {{{info}}} [{local_best_index}] (thread {thread_id})" % (best_score.value,))

                # Save inputs
                with best_inputs.get_lock():
                    best_inputs[:] = test_inputs
                new_best_flag.value = True
                while new_best_flag.value:
                    time.sleep(0.01)
                with best_index.get_lock():
                   best_index.value = local_best_index

                test_inputs, cur_inputs = cur_inputs, test_inputs
                cur_score = score
            elif ((score + SCORE_LENIENCY) > best_score.value):
                if (np.random.random() < np.exp((cur_score - score) / temp)):
                    test_inputs, cur_inputs = cur_inputs, test_inputs
                    cur_score = score

        temp *= TEMP_DECAY_RATE
        with frame_count.get_lock():
            frame_count.value += cycle_frames
                

def printInfo(fmt, val, weight, force=False):
    if (force or (weight != 0)):
        return [fmt % val]
    return []

# Get info string printed with each new best
def getPrintInfo():
    infoStrs = printInfo("X: %.3f", pos[0], POS_WEIGHTS[0], force=False)
    infoStrs += printInfo("Y: %.3f", pos[1], POS_WEIGHTS[1], force=False)
    infoStrs += printInfo("Z: %.3f", pos[2], POS_WEIGHTS[2], force=False)
    infoStrs += printInfo("X Vel: %.3f", vel[0], VEL_WEIGHTS[0], force=False)
    infoStrs += printInfo("Y Vel: %.3f", vel[1], VEL_WEIGHTS[1], force=False)
    infoStrs += printInfo("Z Vel: %.3f", vel[2], VEL_WEIGHTS[2], force=False)
    infoStrs += printInfo("HSpd: %.3f", hspd, HSPD_WEIGHT, force=False)
    infoStrs += printInfo("Pitch: %d", face_angle[0], FACE_ANGLE_WEIGHTS[0], force=False)
    infoStrs += printInfo("Yaw: %d", face_angle[1] & 65535, FACE_ANGLE_WEIGHTS[1], force=False)
    infoStrs += printInfo("Roll: %d", face_angle[2], FACE_ANGLE_WEIGHTS[2], force=False)
    infoStrs += printInfo("Pitch Vel: %d", angle_vel[0], ANGLE_VEL_WEIGHTS[0], force=False)
    infoStrs += printInfo("Yaw Vel: %u", angle_vel[1], ANGLE_VEL_WEIGHTS[1], force=False)
    infoStrs += printInfo("Roll Vel: %d", angle_vel[2], ANGLE_VEL_WEIGHTS[2], force=False)
    infoStrs += printInfo("Coins: %d", num_coins, COINS_WEIGHT, force=False)
    infoStrs += printInfo("Act: %s", sm64Tools.actionToStr(action), (TARGET_ACTION == None), force=False)
    return ", ".join(infoStrs)
        

# Start or restart multiprocessing bruteforcer
def bruteforce(i=-1):
    global all_best_inputs
    
    # Change current best according to start
    all_bests_length = len(all_best_inputs)
    cur_best_index = all_bests_length - 1
    if (i > cur_best_index):
        raise ValueError(f"Index {i} exceeds current best index ({cur_best_index}).")
    i %= all_bests_length
    if (i < cur_best_index):
        _ = input(f"WARNING: Starting bruteforcing from index {i} (current best: {cur_best_index})")
    all_best_inputs = all_best_inputs[:i+1]

    bf_inputs = all_best_inputs[i]
    setup_inputs = np.concatenate((inputs[:START_FRAME], bf_inputs))

    best_inputs = mp.Array(ctypes.c_uint32, bf_inputs)
    best_score = mp.Value(ctypes.c_double)
    best_index = mp.Value(ctypes.c_uint32, i)
    setup_complete = mp.Value(ctypes.c_bool, False)
    new_best_flag = mp.Value(ctypes.c_bool, False)
    frame_count = mp.Value(ctypes.c_ulonglong, 0)
    time_start = mp.Value(ctypes.c_double, 0)

    args = {
        "LENGTH": END_FRAME - START_FRAME,
        "setup_inputs": setup_inputs,
        "best_inputs": best_inputs,
        "best_score": best_score,
        "best_index": best_index,
        "start_best_index": i,
        "setup_complete": setup_complete,
        "new_best_flag": new_best_flag,
        "frame_count": frame_count,
        "time_start": time_start
    }
    if STATE_LOADED:
        args["basis_inputs"] = basis_inputs
    g = globals()
    args.update({k: g[k] for k in g.keys() if k.isupper()})

    with ProcessGroup(NUM_THREADS, mpBruteforce, (args,)) as processes:
        try:
            while processes.active:
                processes.print()
                if new_best_flag.value:
                    all_best_inputs.append(np.uint32(best_inputs))
                    new_best_flag.value = False
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    
    if (time_start.value != 0):
        time_end = time.time()
        fps = frame_count.value / (time_end - time_start.value)
        print(f"FPS: {fps}")


# Output current best (or previous best by index)
def out(i=-1, out_name=OUT_NAME, save_end_inputs=SAVE_END_INPUTS):
    l = len(all_best_inputs)
    if (-l <= i < 0):
        i %= l
    input_blocks = [inputs[:START_FRAME], all_best_inputs[i]]
    if (save_end_inputs):
        input_blocks.append(inputs[END_FRAME:])
    inputs_out = np.concatenate(input_blocks)
    
    print(f"Saving m64 {i} to \"{out_name}\"")
    inputTools.saveM64(out_name, inputs_out, header)

    
if (__name__ == "__main__"):
    # Load m64s
    print(f"Loading m64 \"{INP_NAME}\"")
    inputs, header = inputTools.loadM64(INP_NAME)
    all_best_inputs = [inputs[START_FRAME:END_FRAME]]

    STATE_LOADED = header.state_loaded
    if STATE_LOADED:
        print(f"Loading basis m64 \"{BASIS_NAME}\"")
        basis_inputs = inputTools.loadM64(BASIS_NAME)[0]

    bruteforce()
    #out()
