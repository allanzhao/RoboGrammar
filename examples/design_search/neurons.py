import json
from multiprocessing import Array, Process, Value
from time import sleep, time

import numpy as np
import serial
import zmq
from scipy.stats import gamma


class NeuronStream(Process):
    
    # def __init__(self, channels=16, update_freq=15, activation_fn=gamma, activation_params={"a": 1.1, "scale": 5.0}, buffer_ms=9000, episode_length=128):
    def __init__(self, channels=16, update_freq=15, episode_length=128+16, dt=16/240, raw_values_buffer_ms=9000):
        super(Process, self).__init__()
        
        self.update_frequency = update_freq
        self.channels = channels
        self.raw_values_buffer_ms = raw_values_buffer_ms
        self.episode_length = episode_length
        self.dt = dt
        self.sample_rate = 30000
        self.buffer_size = int((raw_values_buffer_ms / 1000) * self.sample_rate)
        
        # self.activation_fn = activation_fn
        # self.activation_params = activation_params
        
        # self.timestamp_min = activation_fn.ppf(0.001, **activation_params)
        # self.timestamp_max = activation_fn.ppf(0.999, **activation_params)
        
        self.activation_timestamps = [[] for _ in range(channels)]
        # self.activation_values = np.zeros(channels)
        self.activation_values = Array('d', [0] * channels)
        self.raw_values_buffer = Array('d', [0] * self.buffer_size * channels)
        self.run_loop = Value("i", 1)
        self.enough_time = Value("i", 0)
        self.buffer_offset = Array("i", [0] * channels)
        
        self.spike_history_len = 20
        self.raw_spikes_buffer = Array("d", [0] * self.spike_history_len * channels)
        self.spike_offsets = Array("i", [0] * channels)
        self.spike_frequency_buffer = Array("d", [0] * episode_length * channels)
        self.spike_frequency_offset = Value("i", 0)
        
    def run(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5556")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        # self.compute_activations()
        self.receive_data()
        
    def receive(self, data_types={"data", "spike"}):
        try:
            message = self.socket.recv_multipart(zmq.NOBLOCK)
        except zmq.error.Again:
            return None, None
        
        header = json.loads(message[1].decode('utf-8'))
        if "data" in data_types and header["type"] == "data":
            body = np.frombuffer(message[2], dtype=np.float32)
            return header, body
        elif "spike" in data_types and header["type"] == "spike":
            body = np.frombuffer(message[2], dtype=np.float32)
            return header, body
        elif "other" in data_types and header["type"] not in {"spike", "data"}:
            return header, body
        return None, None

    def receive_data(self):
        i = 0
        time_prev = time()
        time_start = time()
        while self.run_loop.value == 1:
            header, body = self.receive({"spike"})
            if header is not None and header["type"] == "data":
                channel = header["content"]["channel_num"]
                for sample in range(len(body)):
                    idx = channel * self.buffer_size + ((self.buffer_offset[channel] + sample) % self.buffer_size)
                    self.raw_values_buffer[idx] = body[sample]
                self.buffer_offset[channel] = self.buffer_offset[channel] + len(body)
                self.buffer_offset[channel] = self.buffer_offset[channel] % self.buffer_size
            if header is not None and header["type"] == "spike":
                channel = int(header["spike"]["electrode"].split(" ")[-1]) - 1
                # ^This depends on the electrode names in OpenEphys. Assumes interval [0, 15].
                idx = channel * self.spike_history_len + ((self.spike_offsets[channel] + 1) % self.spike_history_len)
                self.raw_spikes_buffer[idx] = time()
                self.spike_offsets[channel] = self.spike_offsets[channel] + 1
                self.spike_offsets[channel] = self.spike_offsets[channel] % self.spike_history_len
                
            if (time() - time_prev) > (1 / self.update_frequency):
                time_prev = time()
                i += 1
                freqs = self.get_spike_frequencies()
                # print(i, time(), np.all(freqs < 10))
                for channel in range(self.channels):
                    idx = channel * self.episode_length + ((self.spike_frequency_offset.value + 1) % self.episode_length)
                    self.spike_frequency_buffer[idx] = freqs[channel]
                self.spike_frequency_offset.value += 1
                self.spike_frequency_offset.value %= self.episode_length
    
            if self.enough_time.value == 0 and (time() - time_start) > (self.dt * self.episode_length):
                self.enough_time.value = 1
            
    """
    def compute_activations(self):
        update_time = 1 / self.update_frequency
        time_start = time()
        while self.run_loop.value == 1:
            header, body = self.receive({"spike"})
            if header is not None and header["type"] == "spike":
                assert header["spike"]["n_channels"] == 1
                electrode_id = header["spike"]["electrode_id"]
                self.activation_timestamps[electrode_id].append(time())
            
            time_end = time()
            if (time_end - time_start) >= update_time:
                for electrode_id, timestamps in enumerate(self.activation_timestamps):
                    electrode_value = 0
                    to_delete = set()
                    for timestamp_i, timestamp in enumerate(timestamps):
                        timestamp_diff = time_end - timestamp
                        if self.timestamp_min < timestamp_diff <= self.timestamp_max:
                            electrode_value += self.activation_fn.pdf(timestamp_diff, **self.activation_params)
                        elif timestamp_diff > self.timestamp_max:
                            to_delete.add(timestamp_i)
                            
                    self.activation_values[electrode_id] = electrode_value
                    self.activation_timestamps[electrode_id] = [timestamp for timestamp_i, timestamp in enumerate(timestamps) if timestamp_i not in to_delete]
                    
                ### STATS PRINTING
                # print(["%.2f" % val for val in self.activation_values], end="\r")
                # print(["%.2f" % val for val in self.activation_derivatives], end="\r")
                # print(["%d" % len(value) for key, value in self.activation_timestamps.items()], end="\r")  
                # print(time_end - time_start, end="\r")
                
                time_start = time()
    
    """      
                
    def stop(self):
        print("Stopping neuron stream")
        self.run_loop.value = 0
        sleep(1)

    def get_raw_values_array(self):
        return_array = np.zeros((self.channels, self.buffer_size))
        # print(self.buffer_size)
        for channel in range(self.channels):
            for sample in range(self.buffer_size):
                idx = channel * self.buffer_size + ((self.buffer_offset[channel] + sample) % self.buffer_size)
                return_array[channel, sample] = self.raw_values_buffer[idx]
        return return_array
    
    def get_spike_frequencies(self, channels=None):
        if channels is None:
            channels = np.arange(self.channels)
        frequencies = np.zeros(self.channels)
        
        for channel in channels:
            channel_events = []
            for event_idx in range(self.spike_history_len):
                idx = channel * self.spike_history_len + ((self.spike_offsets[channel] + event_idx) % self.spike_history_len)
                channel_events.append(self.raw_spikes_buffer[idx])
            channel_events = np.array(channel_events)
            channel_events = channel_events[channel_events != 0]
            if len(channel_events) <= 1:
                frequencies[channel] = 0
            else:
                time_delta = channel_events.max() - channel_events.min()
                frequencies[channel] = len(channel_events) / time_delta
        
        return frequencies

    def get_spike_frequencies_array(self):
        while self.enough_time.value == 0:
            sleep(0.1)
            print("Waiting to gather data.", end="\r")
        if self.enough_time.value == 1:
            print(" " * 30, end="\r")
            
        return_array = np.zeros((self.channels, self.episode_length))
        for channel in range(self.channels):
            for sample in range(self.episode_length):
                idx = channel * self.episode_length + ((self.spike_frequency_offset.value + sample) % self.episode_length)
                return_array[channel, sample] = self.spike_frequency_buffer[idx]
        return return_array


class NeuronStreamWrapper:
    
    def __init__(self, weights=None, neuron_stream_std_multiplier=8, stream_kwargs={}):
        self.neuron_stream = NeuronStream(**stream_kwargs)
        self.last_neuron_readout = None
        self.neuron_stream_std_multiplier = neuron_stream_std_multiplier
        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.diag(np.ones(self.neuron_stream.channels))
            # rng = np.random.RandomState(0)
            # self.weights = rng.rand(self.neuron_stream.channels, dof_count)
            # self.weights = self.weights / self.weights.sum(axis=1).reshape((-1, 1))
        
    def start(self):
        self.neuron_stream.start()
        sleep(self.neuron_stream.episode_length * self.neuron_stream.dt / 1000)
        
    def stop(self):
        self.neuron_stream.stop()
        
    def get_channel_frequencies(self, most_current=True):
        if not most_current or self.last_neuron_readout is None:
            self.last_neuron_readout = self.neuron_stream.get_spike_frequencies_array()
        return self.last_neuron_readout


if __name__ == "__main__":
    try:
        neurons = NeuronStream(channels=32)
        neurons.start()
        
        sleep(1)
        frequencies = []
        timestamps = []
        time_start = time()
        while time() - time_start < 10:
            # n = neurons.get_raw_values_array()
            # print(n, n.shape)
            # frequencies.append(neurons.get_spike_frequencies()[:16])
            # timestamps.append(time())
            sleep(0.05)
            # print()
            # print(n[0, :])
            # raise KeyboardInterrupt
        
        from matplotlib import pyplot as plt
        # frequencies = np.stack(frequencies, axis=-1)
        frequencies = neurons.get_spike_frequencies_array()[:16]
        # frequencies = frequencies.clip(0, 100)
        # timestamps = np.array(timestamps) - min(timestamps)
        timestamps = np.linspace(0, 16/240*neurons.episode_length, neurons.episode_length)
        # timestamps = np.stack([timestamps] * 16, axis=0)
        print(frequencies.shape, timestamps.shape)
        fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(16, 16))
        axes = np.reshape(axes, -1)
        for i in range(len(frequencies)):
            axes[i].plot(timestamps, frequencies[i], label=str(i))
            axes[i].set_title("channel=" + str(i))
        # plt.legend()
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("frequency")
        
        plt.tight_layout()
        plt.savefig("freqs.png")
        
    except KeyboardInterrupt:
        print()
        neurons.stop()
            
    except Exception as e:
        print(e)