import matplotlib.pyplot as plt
import numpy as np
import socket
import struct
import time

from CONFIGURATION import SERVER_PORT, SAMPLING_RATE, DATA_PACKET_FORMAT

# Configuration
SAMPLE_WINDOW_SIZE = 25 # samples
CHANNEL_COLORS = ['r', 'g', 'b', 'y', 'm', 'c']
CHANNEL_NAMES = ["ACC X", "ACC Y"]
DATA_OFFSET = 510

# Constants
NUM_CHANNELS = len(CHANNEL_NAMES)
CHANNEL_COLORS = CHANNEL_COLORS[0:NUM_CHANNELS]
TIMESTEP = 1 / SAMPLING_RATE
ADC_MAX = 1024

plt.ion()
fig, ax = plt.subplots()
channel_lines = [ax.plot([], [], f'{color}-', label=CHANNEL_NAMES[i])[0] for i, color in enumerate(CHANNEL_COLORS)]
plt.title("Real-time signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, ADC_MAX)
xdata = []
channel_ydatas = [[] for _ in range(NUM_CHANNELS)]

plot_active = True
def on_plot_close(event):
    global plot_active
    plot_active = False
fig.canvas.mpl_connect('close_event', lambda event: on_plot_close(event))

# new_channels_sample: list of one sample per channel
def add_sampling_to_plot(sampling_time, new_channels_sample):
    xdata.append(sampling_time)
    for i, channel_sample in enumerate(new_channels_sample):
        channel_ydatas[i].append(channel_sample)

    # found empirically
    wnd_xmin = 0 if len(xdata) < SAMPLE_WINDOW_SIZE else xdata[-SAMPLE_WINDOW_SIZE]
    wnd_xmax = 3 if len(xdata) < SAMPLE_WINDOW_SIZE else xdata[-1]
    ax.set_xlim(wnd_xmin, wnd_xmax)

    if len(xdata) >= SAMPLE_WINDOW_SIZE:
        recent_samples = [channel[-SAMPLE_WINDOW_SIZE:] for channel in channel_ydatas]
    else:
        recent_samples = channel_ydatas

    all_recent_samples = [sample for sublist in recent_samples for sample in sublist]
    min_sample = min(all_recent_samples)
    max_sample = max(all_recent_samples)
    delta = max_sample - min_sample
    ax.set_ylim(min_sample - 0.1 * delta, max_sample + 0.1 * delta)

    for i in range(len(new_channels_sample)):
        channel_lines[i].set_xdata(xdata)
        channel_lines[i].set_ydata(channel_ydatas[i])

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', SERVER_PORT))

def receive_samples():
    # Send a byte to the server to signal we're ready to receive samples
    client_socket.sendall(b'\x00')

    # Receive the samples (always six 16-bit values)
    serialized_data = client_socket.recv(6 * 2)

    # Unpack the samples
    return [x - DATA_OFFSET for x in list(DATA_PACKET_FORMAT.unpack(serialized_data))][:NUM_CHANNELS]

try:
    start_time = time.time()
    while plot_active:
        samples = receive_samples()
        delta = time.time() - start_time
        add_sampling_to_plot(delta, samples)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

finally:
    client_socket.close()
