from collections import deque

import itertools
import matplotlib.pyplot as plt
import numpy as np
import socket
import time

# import our own libraries
import utility.FilterToolkit as uFT


from CONFIGURATION import SERVER_PORT, SAMPLING_RATE

# Configuration
SAMPLE_WINDOW_SIZE = 25 # samples

class Channel(deque):
    """
    Represents a single channel of data samples.

    When appending a new sample, the mean value will automatically
    be subtracted from it, which is useful to eliminate a DC offset.
    Optionally, the data can also be filtered using a Filter object.
    """
    def __init__(self,
                 name: str,
                 color: str,
                 window_size: int,
                 filter: uFT.Filter | None = None):
        super().__init__(maxlen=window_size)
        self.name = name
        self.color = color
        self.mean = 0
        self._filter = filter

    #@overload
    def append(self, e: int) -> None:
        e -= self.mean
        if self._filter is not None:
            e = self._filter(e)
        super().append(e)

    def data__for_graph_only(self) -> list[int]:
        return [x for x in self]

    def all_outside_range(self, threshold: int) -> bool:
        return all(sample > threshold for sample in self)

class DynamicPlotManager:
    def __init__(self,
                 title: str,
                 channels: list[Channel],
                 window_size: int,
                 y_window: tuple[int, int] | None = None):
        assert(all(ch.maxlen == window_size for ch in channels))

        self.window_size = window_size
        self.y_window = y_window

        self.fig, self.ax = plt.subplots()
        self.plot_lines = [self.ax.plot([], [], f'{ch.color}-', label=ch.name)[0] for ch in channels]

        self.active = True
        self.user_elements: dict[str] = {}
        self.start_timestamp = time.time()

        self.xdata = deque(maxlen=window_size)

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend(loc='center left')
        plt.ion()

        if self.y_window is not None:
            self.ax.set_ylim(self.y_window[0], self.y_window[1])

        def on_plot_close(event):
            self.active = False
        self.fig.canvas.mpl_connect('close_event', lambda event: on_plot_close(event))

    def _recompute_wnd_limits(self, channels: list[Channel]) -> None:
        if len(self.xdata) < self.window_size:
            xmin = 0
            xmax = 3
        else:
            xmin = self.xdata[-self.window_size]
            xmax = self.xdata[-1]
        self.ax.set_xlim(xmin, xmax)

        # Only update Y limits if not set by user
        if self.y_window is None:
            ymin = min([min(ch) for ch in channels])
            ymax = max([max(ch) for ch in channels])
            ydelta = ymax - ymin
            y_margin = ydelta * 0.1

            self.ax.set_ylim(ymin - y_margin, ymax + y_margin)

    def add_tick(self) -> None:
        self.xdata.append(time.time() - self.start_timestamp)

    def reset(self) -> None:
        self.xdata.clear()
        self.start_timestamp = time.time()

    def redraw(self, channels: list[Channel], additional: dict[str, list[int]] | None = None) -> None:
        # update window limits
        self._recompute_wnd_limits(channels)

        # update all lines with new data
        for line, channel in zip(self.plot_lines, channels):
            line.set_data(self.xdata, channel.data__for_graph_only())

        # update additional user plots (elements with "plot_" prefix)
        if additional is not None:
            for name, element in self.user_elements.items():
                if not name.startswith("plot_"):
                    continue

                data_name = name[5:] # remove "plot_" prefix

                if additional.get(data_name) is not None:
                    element.set_data(self.xdata, additional[data_name])

        # perform draw and handle window events
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def __setitem__(self, name: str, element):
        if self.user_elements.get(name) is not None:
            raise ValueError(f"Element with name {name} already exists.")
        self.user_elements[name] = element

    def __getitem__(self, name: str):
        return self.user_elements.get(name)

class SamplingClient:
    def __init__(self, server_port: int, num_channels: int):
        self.num_channels = num_channels
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(('localhost', server_port))

    def receive_samples(self) -> list[int]:
        # Send a byte to the server to signal we're ready to receive samples
        self._sock.sendall(b'\x00')

        # Receive the samples (always six 16-bit values)
        serialized_data = self._sock.recv(6 * 2)

        # Unpack the samples
        from CONFIGURATION import DATA_PACKET_FORMAT as dpf
        return list(dpf.unpack(serialized_data))[:self.num_channels]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._sock.close()

class PhyMice:
    def __init__(self,
                 channels_config: dict[str, dict],
                 window_size: int,
                 sampling_rate: int):
        self.sampling_rate = sampling_rate
        self.sampling_client: SamplingClient = None

        self.channels = { name: Channel(name, cfg["color"], window_size, cfg.get("filter")) for name, cfg in channels_config.items() }
        self.channels_state: dict[str, int] = { name: 0 for name in self.channels.keys() }
        self.idle_sensitivities = { name: cfg["idle_sensitivity"] for name, cfg in channels_config.items() }
        self.active_sensitivities = { name: cfg.get("active_sensitivity", 0) for name, cfg in channels_config.items() }

        self.plotmgr = DynamicPlotManager("Real-time signals", self.channels.values(), window_size, y_window=(-100, 100))
        self.plotmgr["unstable_msg"] = (
            self.plotmgr.fig.text(1, 1, "UNSTABLE", fontsize=10, color='red',
                horizontalalignment='right', transform=self.plotmgr.ax.transAxes))

#        self.plotmgr["emg_derive"] = self.plotmgr.ax.plot([], [], 'y-', label="EMG Derivative")[0]

    def _perform_calibration(self) -> bool:
        print("Performing calibration... Please keep the device still.")

        calibration_samples = []
        for _ in range(self.sampling_rate // 4): # calibrate over 0.25 seconds
            samples = self.sampling_client.receive_samples()
            calibration_samples.append(samples)

        calibration_means = [sum(x) / len(calibration_samples) for x in zip(*calibration_samples)]
        calibration_stds = [np.std([sample[i] for sample in calibration_samples]) for i in range(len(calibration_samples[0]))]

        has_unstable = False
        for (name, ch), std in zip(self.channels.items(), calibration_stds):
            if self.idle_sensitivities[name] != 0 and std >= self.idle_sensitivities[name]:
                print(f"{ch.name}: too much variation during calib! ({std=:0.2f} > {self.idle_sensitivities[name]})")
                has_unstable = True

        if has_unstable:
            return False

        for ch, calib in zip(self.channels.values(), calibration_means):
            ch.mean = int(calib)
        return True

    def _unstable(self) -> bool:
        unstable = False
        for name, ch in self.channels.items():
            if (name != "Force"
                and self.idle_sensitivities[name] != 0
                and ch.all_outside_range(self.idle_sensitivities[name])
               ):
                unstable = True
                break
        return unstable

    def _process_signals(self):
        acc_x = self.channels["Acc X"]
        acc_y = self.channels["Acc Y"]
        emg = self.channels["EMG"]
        force = self.channels["Force"]

        active = (force[-1] > 20) # threshold doesn't really matter
        inhibited = '' if active else '(inhibited)'

        # Attempt to detect movement on X and Y axis
        for ch in [acc_x, acc_y]:
            # if unstable, don't process channel
            if ch.all_outside_range(self.idle_sensitivities[ch.name]):
                # todo: ch_state[i] = 0?
                continue

            idle_thr = self.idle_sensitivities[ch.name]
            act_thr = self.active_sensitivities[ch.name]
            cur_state = self.channels_state[ch.name]
            # depends on ACC orientation
            DIRECTIONS = {
                True: "left",
                False: "right",
            } if ch.name == "Acc X" else {
                True: "up",
                False: "down",
            }

            def _idle(i):
                return -idle_thr <= ch[i] <= idle_thr

            def _active(i):
                return not -act_thr <= ch[i] <= act_thr

            if cur_state == 0: # idle
                # transition to active if the last few samples are active
                if _active(-1) and _active(-2) and _active(-3):
                    positive_move = (ch[-3] - ch[-2]) < 0
                    print(f"Move {DIRECTIONS[positive_move]} {inhibited}")
                    self.channels_state[ch.name] = 1

            elif cur_state == 1: # moving
                # transition to idle if either the last 5 samples are idle;
                # or the last 2 samples are idle and previous was active
                if _active(-3) and all(_idle(-i) for i in range(1, 2+1)) \
                    or all(_idle(-i) for i in range(1, 5+1)):
                    self.channels_state[ch.name] = 0

        # Attempt to detect muscle activity in EMG
        # (active if the delta between max and min is
        #  above threshold for the past 5 samples)
        if len(emg) >= 5:
            emg_wnd = [emg[i] for i in range(-5, 0)]
            emg_delta = max(emg_wnd) - min(emg_wnd)
            if self.channels_state[emg.name] == 0:
                if emg_delta > self.active_sensitivities[emg.name]:
                    print(f"Muscle activity detected! {inhibited}")
                    self.channels_state[emg.name] = 1
            elif emg_delta <= self.idle_sensitivities[emg.name]:
                    self.channels_state[emg.name] = 0

        return active

    def go(self):
        RETRY_INTERVAL = 3

        with SamplingClient(SERVER_PORT, len(self.channels)) as self.sampling_client:
            while not self._perform_calibration():
                print(f"Calibration failed. Trying again in {RETRY_INTERVAL} seconds...")
                time.sleep(RETRY_INTERVAL)

            print("Calibration successful:")
            for ch in self.channels.values():
                print(f"\t{ch.name}: mean={ch.mean}")

            print("Starting movement detection...")

            self.plotmgr.reset()
            self.plotmgr.fig.show()
            while self.plotmgr.active:
                self.plotmgr.add_tick()
                samples = self.sampling_client.receive_samples()
                for ch, sample in zip(self.channels.values(), samples):
                    ch.append(sample)

                self.plotmgr["unstable_msg"].set_visible(self._unstable())

                self._process_signals()

                # calculate EMG derivative
#                emg = self.channels[3]
#                emg_derive = [0] + [emg[i] - emg[i-1] for i in range(1, len(emg))]

                self.plotmgr.redraw(self.channels.values())

def main():
    cfg = {
        "Acc X": {
            "color": "r",
            "idle_sensitivity": 3,
            "active_sensitivity": 10,
            "filter": uFT.ButterworthLPF(cutoff=3, fs=20, order=2)
        },
        "Acc Y": {
            "color": "b",
            "idle_sensitivity": 3,
            "active_sensitivity": 10,
            "filter": uFT.ButterworthLPF(cutoff=3, fs=20, order=2)
        },
        "Force": {
            "color": "c",
            "idle_sensitivity": 0
        },
        "EMG": {
            "color": "m",
            "idle_sensitivity": 25,
            "active_sensitivity": 60,
#            "filter": uFT.ButterworthLPF(cutoff=2.5, fs=20, order=2)
        }
    }

    PhyMice(cfg, SAMPLE_WINDOW_SIZE, SAMPLING_RATE).go()

if __name__ == "__main__":
    main()
