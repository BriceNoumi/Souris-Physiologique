import struct

# Configuration
MAC_ADDRESS = "98:D3:51:FE:87:0B"
SAMPLING_RATE = 1000 # Hz
CHANNELS = [0, 1, 2, 3, 4, 5]
SERVER_PORT = 60425

DATA_PACKET_FORMAT = struct.Struct(">6H")