import numpy as _np
import os as _os
import pickle as _pkl
import wafel as _wafel

_ = _np.seterr(over="ignore", under="ignore")
_core_path = _os.path.dirname(__file__)


# Actions
with open(_os.path.join(_core_path, "data/actions.pkl"), "rb") as _fp:
    _act_strs = _pkl.load(_fp)
_valid_acts = _act_strs.keys()

actionToStr = lambda a: _act_strs[a] if a in _valid_acts else "unknown action"


# RNG
def rngIncr(v):
    if (v == 0x560A):
        v = 0
    s0 = (((v & 0xFF) << 8) ^ v)
    v = (((s0 & 0xFF) << 8) | (s0 >> 8))
    s0 = (((s0 & 0xFF) << 1) ^ v)
    s1 = ((s0 >> 1) ^ 0xFF80)
    if (s0 & 1):
        return (s1 ^ 0x8180)
    elif (s1 == 0xAA55):
        return 0
    return (s1 ^ 0x1FF4)

with open(_os.path.join(_core_path, "data/rngData.pkl"), "rb") as _fp:
    rng_indices, rng_values = _pkl.load(_fp)

getRngIndex = lambda game: rng_indices[game.read("gRandomSeed16")]
setRngIndex = lambda game, rng_index: game.write("gRandomSeed16", int(rng_values[rng_index]))


# Trig
with open(_os.path.join(_core_path, "data/trigTables.pkl"), "rb") as _fp:
    _sin_table, _arctan_table = _pkl.load(_fp)

def sins(x):
    return _sin_table[(x >> 4) & 0xFFF]

def coss(x):
    return _sin_table[((x + 16384) >> 4) & 0xFFF]

def _atan2_lookup(y, x):
    if (x == _np.float32(0)):
        return _arctan_table[0]
    return _arctan_table[_np.int32(y / x * 1024 + 0.5)]

def atan2s(y, x):
    x = _np.float32(x)
    y = _np.float32(y)
    if (x >= 0):
        if (y >= 0):
            if (y >= x):
                return _atan2_lookup(x, y)
            return 0x4000 - _atan2_lookup(y, x)
        y = -y
        if (y < x):
            return 0x4000 + _atan2_lookup(y, x)
        return 0x8000 - _atan2_lookup(x, y)
    x = -x
    if (y < 0):
        y = -y
        if (y >= x):
            return 0x8000 + _atan2_lookup(x, y)
        return 0xC000 - _atan2_lookup(y, x)
    if (y < x):
        return 0xC000 + _atan2_lookup(y, x)
    return -_atan2_lookup(x, y)


# Game read/write
def getValSafe(game, var_name, default=0):
    try:
        return game.read(var_name)
    except _wafel.WafelError:
        return default

def setValSafe(game, var_name, val):
    try:
        game.write(var_name, val)
    except _wafel.WafelError:
        pass

def getAddress(wafel_addr):
    if wafel_addr.is_null():
        return 0
    return int(str(wafel_addr)[11:-1], 16) # Stupid hacky work-around


# Objects
def enumSlots(game, processing_groups=(11, 9, 10, 0, 5, 4, 2, 6, 8, 12)):
    OBJ_POOL_START_ADDR = getAddress(game.address("gObjectPool"))
    OBJ_SIZE = (getAddress(game.address("gObjectPool[1]")) - OBJ_POOL_START_ADDR)
    OBJ_POOL_END_ADDR = (OBJ_POOL_START_ADDR + (OBJ_SIZE * 240))

    for processing_group in processing_groups:
        start_str = f"gObjectListArray[{processing_group}]"
        if game.address(start_str).is_null():
            continue
        
        obj_address = getAddress(game.read(f"{start_str}.next"))
        while ((obj_address >= OBJ_POOL_START_ADDR) and (obj_address < OBJ_POOL_END_ADDR)):
            slot = int((obj_address - OBJ_POOL_START_ADDR) // OBJ_SIZE)
            yield slot
            obj_address = getAddress(game.read(f"gObjectPool[{slot}].header.next"))
            
getObjPos = lambda game, slot: game.read(f"gObjectPool[{slot}].rawData.asF32")[6:9]
getObjAct = lambda game, slot: game.read(f"gObjectPool[{slot}].oAction")
getObjTimer = lambda game, slot: game.read(f"gObjectPool[{slot}].oTimer")
getObjBhv = lambda game, slot: game.read(f"gObjectPool[{slot}].behavior")

getMarioObjTimer = lambda game: getValSafe(game, "gMarioObject->oTimer")

# Coin info
objIsCoin = lambda game, slot: (game.read(f"gObjectPool[{slot}].oInteractType") == 0x10)
getCoinValue = lambda game, slot: game.read(f"gObjectPool[{slot}].oDamageOrCoinValue")
getCoinRadius = lambda game, slot: game.read(f"gObjectPool[{slot}].hitboxRadius")

getCoinTangibility = lambda game, slot: (game.read(f"gObjectPool[{slot}].oIntangibleTimer") == 0)
setCoinTangibility = lambda game, slot, tangible: game.write(f"gObjectPool[{slot}].oIntangibleTimer", 0 if tangible else -1)


# Game checks
getFlags = lambda game, var, flags: ((game.read(var) & flags) == flags)
def setFlags(game, var, flags, on):
    cur_flags = game.read(var)
    if on:
        new_flags = cur_flags | flags
    else:
        new_flags = cur_flags & ~flags
    if (cur_flags != new_flags):
        game.write(var, new_flags)

getSaveFileNum = lambda game: max(1, game.read("gCurrSaveFileNum")) - 1

getWingCap = lambda game: getFlags(game, "gMarioState.flags", 8)
getMetalCap = lambda game: getFlags(game, "gMarioState.flags", 4)
getVanishCap = lambda game: getFlags(game, "gMarioState.flags", 2)
setWingCap = lambda game, has_cap: setFlags(game, "gMarioState.flags", 8, has_cap)
setMetalCap = lambda game, has_cap: setFlags(game, "gMarioState.flags", 4, has_cap)
setVanishCap = lambda game, has_cap: setFlags(game, "gMarioState.flags", 2, has_cap)

getWingCapSwitch = lambda game: getFlags(game, "gSaveBuffer.files[%d][0].flags" % getSaveFileNum(game), 2)
getMetalCapSwitch = lambda game: getFlags(game, "gSaveBuffer.files[%d][0].flags" % getSaveFileNum(game), 4)
getVanishCapSwitch = lambda game: getFlags(game, "gSaveBuffer.files[%d][0].flags" % getSaveFileNum(game), 8)
setWingCapSwitch = lambda game, is_pressed: setFlags(game, "gSaveBuffer.files[%d][0].flags" % getSaveFileNum(game), 2, is_pressed)
setMetalCapSwitch = lambda game, is_pressed: setFlags(game, "gSaveBuffer.files[%d][0].flags" % getSaveFileNum(game), 4, is_pressed)
setVanishCapSwitch = lambda game, is_pressed: setFlags(game, "gSaveBuffer.files[%d][0].flags" % getSaveFileNum(game), 8, is_pressed)

def checkDoubleStarSpawn(game):
    star_behaviors = (game.address("bhvSpawnedStar"), game.address("bhvStarSpawnCoordinates"), game.address("bhvSpawnedStarNoLevelExit"))
    count = 0
    for slot in enumSlots(game, processing_groups=(6,)):
        cur_bhv = getObjBhv(game, slot)
        if cur_bhv not in star_behaviors:
            continue
        if (getObjAct(game, slot) < 2):
            count += 1
    return (count >= 2)

getGlobalTimer = lambda game: game.read("gGlobalTimer")
setGlobalTimer = lambda game, val: game.write("gGlobalTimer", val)


# Camera
getCamYaw = lambda game: getValSafe(game, "gMarioState.area->camera->yaw")
setCamYaw = lambda game, yaw: setValSafe(game, "gMarioState.area->camera->yaw", yaw)

isFixedCamMode = lambda game: ((game.read("sSelectionFlags") & 4) == 0)
def setFixedCamMode(game, use_fixed_cam):
    selection_flags = game.read("sSelectionFlags")
    is_using_fixed_cam = ((selection_flags & 4) == 0)
    if (is_using_fixed_cam != use_fixed_cam):
        game.write("sSelectionFlags", selection_flags ^ 4)
