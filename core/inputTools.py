import numpy as _np
import os as _os
import pickle as _pkl
import struct as _struct


# Convert to and from int input and split stick/button inputs
joinInputs = lambda stick_x, stick_y, buttons: (buttons << 16) | ((stick_x & 0xFF) << 8) | (stick_y & 0xFF)
def splitInputs(inputs):
    inputs = int(inputs)
    stick_x = (inputs >> 8) & 0xFF
    stick_y = inputs & 0xFF
    buttons = inputs >> 16
    return (stick_x, stick_y, buttons)


# Advance frame with inputs
def advanceFrame(game, inputs):
    stick_x, stick_y, buttons = splitInputs(inputs)
    game.write("gControllerPads[0].stick_x", stick_x)
    game.write("gControllerPads[0].stick_y", stick_y)
    game.write("gControllerPads[0].button", buttons)
    game.advance()


# M64 header
class M64Header:
    def __init__(self, data):
        if (len(data) != 1024):
            raise ValueError("Header data incorrect size (should be 0x400)")

        header_signature = data[:4]
        if (header_signature != b"M64\x1a"):
            raise ValueError("Unexpected header signature")

        header_version = data[4:8]
        if (header_version != b"\x03\x00\x00\x00"):
            raise ValueError("Unexpected header version")

        self.__data = bytearray(data)
        self.__data[0x27] = 45

    def toBytes(self):
        return bytes(self.__data)

    def copy(self):
        return M64Header(self.__data.copy())


    def __repr__(self):
        rom_version = self.rom_version
        uid = self.uid
        author = self.author
        description = self.description
        rerecord_count = self.rerecord_count
        state_loaded = self.state_loaded
        return f"""M64Header(
    rom_version = {rom_version},
    uid = {uid},
    author = \"{author}\",
    description = \"{description}\",
    rerecord_count = {rerecord_count},
    state_loaded = {state_loaded}
)"""
    

    __getProperty = lambda self, offset, size, prop_type: _struct.unpack(prop_type, self.__data[offset:offset+size])[0]
    def __setProperty(self, val, offset, size, prop_type):
        val &= (1 << size*8) - 1
        self.__data[offset:offset+size] = _struct.pack(prop_type, val)

    __getString = lambda self, offset, size: self.__data[offset:offset+size].decode("utf-8").split("\0")[0]
    def __setString(self, val, offset, size):
        val_bytes = val.encode("utf-8")
        l = len(val_bytes)
        if (l > size):
            raise ValueError(f"String too long (max size={size})")
        self.__data[offset:offset+l] = val_bytes
        self.__data[offset+l:offset+size] = bytearray(size - l)

    __property = lambda offset, size, prop_type: property(
        fget = lambda self: self.__getProperty(offset, size, prop_type),
        fset = lambda self, val: self.__setProperty(val, offset, size, prop_type)
    )
    __stringProperty = lambda offset, size: property(
        fget = lambda self: self.__getString(offset, size),
        fset = lambda self, val: self.__setString(val, offset, size)
    )
    

    def __getRerecordCount(self):
        rerecords = self.__getProperty(0x010, 4, "I")
        if self.extended_version:
            rerecords |= self.__getProperty(0x02C, 4, "I") << 32
        return rerecords

    def __setRerecordCount(self, val):
        self.__setProperty(val, 0x10, 4, "I")
        if self.extended_version:
            self.__setProperty(val >> 32, 0x2C, 4, "I")

    __getStateLoaded = lambda self: ((self.__getProperty(0x01C, 2, "H") & 1) == 1)
    def __setStateLoaded(self, val):
        start_flags = self.__getProperty(0x01C, 2, "H") & 0xFFFC
        if val:
            start_flags |= 1
        else:
            start_flags |= 2
        self.__setProperty(start_flags, 0x01C, 2, "H")

    __versions = {
        0xFF2B5A63: "US",
        0x0E3DAA4E: "JP",
        0x36F03CA0: "PAL",
        0xA8A4FBD6: "SH"
    }
    def __getVersion(self):
        crc_code = self.__getProperty(0x0E4, 4, "I")
        if crc_code not in self.__versions.keys():
            raise ValueError("Invalid crc code")
        return self.__versions[crc_code]

    __version_codes = (
        (("US", "U"), 0xFF2B5A63, 0x0045),
        (("JP", "J"), 0x0E3DAA4E, 0x004A),
        (("PAL", "P", "EU", "E"), 0x36F03CA0, 0x0050),
        (("SH", "S"), 0xA8A4FBD6, 0x034A)
    )
    def __setVersion(self, val):
        val = val.upper()
        for versions, crc_code, country_code in self.__version_codes:
            if val in versions:
                self.__setProperty(crc_code, 0x0E4, 4, "I")
                self.__setProperty(country_code, 0x0E8, 2, "H")
                return
        raise ValueError("Invalid version")
        

    extended_version = property(
        fget = lambda self: self.__getProperty(0x016, 1, "?")
    )

    rerecord_count = property(
        fget = __getRerecordCount,
        fset = __setRerecordCount
    )

    state_loaded = property(
        fget = __getStateLoaded,
        fset = __setStateLoaded
    )

    rom_version = property(
        fget = __getVersion,
        fset = __setVersion
    )

    uid = __property(0x008, 4, "I")
    length_vis = __property(0x00C, 4, "I")
    length_inputs = __property(0x018, 4, "I")

    rom_name = __stringProperty(0x0C4, 32)
    video_plugin = __stringProperty(0x122, 64)
    audio_plugin = __stringProperty(0x162, 64)
    input_plugin = __stringProperty(0x1A2, 64)
    rsp_plugin = __stringProperty(0x1E2, 64)
    author = __stringProperty(0x222, 222)
    description = __stringProperty(0x300, 256)


# Load m64
def loadM64(filename):
    with open(filename, "rb") as fp:
        header_bytes = fp.read(0x400)
        inputs_file = _np.fromfile(fp, dtype=">u4")

    inputs = inputs_file.astype(_np.uint32)
    header = M64Header(header_bytes)
    return (inputs, header)

# Save m64
def saveM64(filename, inputs, header, padding_frames=1000):
    header_cur = header.copy()
    header_cur.length_inputs = inputs.size + padding_frames
    header_cur.length_vis = -1
    
    header_bytes = header_cur.toBytes()
    inputs_file = inputs.astype(">u4")
    padding_bytes = bytes(padding_frames*4)
    
    with open(filename, "wb") as fp:
        fp.write(header_bytes)
        inputs_file.tofile(fp)
        fp.write(padding_bytes)


# Joystick solving
with open(_os.path.join(_os.path.dirname(__file__), "data/stickData.pkl"), "rb") as _fp:
    _stick_data, _fstick_data, _mag_data, _yaw_data, _u_yawmag_indices = _pkl.load(_fp)
_u_yaws = _yaw_data[_u_yawmag_indices]
_u_mags = _mag_data[_u_yawmag_indices]

def getProcessedJoystick(raw_x, raw_y):
    if (raw_x >= 8):
        stick_x = raw_x - 6
    elif (raw_x <= -8):
        stick_x = raw_x + 6
    else:
        stick_x = 0
    if (raw_y >= 8):
        stick_y = raw_y - 6
    elif (raw_y <= -8):
        stick_y = raw_y + 6
    else:
        stick_y = 0
    return (stick_x, stick_y)

def getNearestJoysticksRaw(base_joystick, n=58564):
    if (n >= 58564):
        n = 58564

    base_joystick = _np.int8(base_joystick)
    raw_joystick = getProcessedJoystick(*base_joystick)
    joystick_dists = _np.linalg.norm(_fstick_data - raw_joystick, axis=1)

    # Get n closest raw joysticks sorted
    best_inds = _np.argpartition(joystick_dists, n-1)[:n]
    subset_inds = joystick_dists[best_inds].argsort()
    return _stick_data[best_inds[subset_inds]]

def getNearestJoysticks(mag, yaw, n=20129, mag_weight=1, yaw_weight=0.0001):
    if (n >= 20129):
        n = 20129
        
    mag_dists = _u_mags - mag
    yaw_dists = _np.int16(_u_yaws - yaw)
    yaw_dists[10064] = 0 # Ignore yaw dist on 0 input
    joystick_dists = mag_weight*_np.square(mag_dists) + yaw_weight*_np.square(_np.float32(yaw_dists))

    # Get n closest raw joysticks sorted
    best_inds = _np.argpartition(joystick_dists, n-1)[:n]
    subset_inds = joystick_dists[best_inds].argsort()
    return _stick_data[_u_yawmag_indices[best_inds[subset_inds]]]
