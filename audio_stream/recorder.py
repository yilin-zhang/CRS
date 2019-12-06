import pyaudio
import wave
import numpy as np
import math
from termcolor import colored, cprint

class Recorder(object):
    '''A recorder class for recording audio to a WAV file.
    Records in mono by default.
    '''

    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname, input_device_index=0, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                            input_device_index, self.frames_per_buffer)

class RecordingFile(object):
    def __init__(self, fname, mode, channels, 
                rate, input_device_index, frames_per_buffer):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None
        self.input_device_index = input_device_index

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        input_device_index = self.input_device_index,
                                        frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        output=True,
                                        input_device_index = self.input_device_index,
                                        frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self.get_callback())
        self._stream.start_stream()

        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self
    
    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            # TODO: This variable should be used in GUI
            volume = self._get_volume(in_data)
            self._visualize_volume(volume)
            return in_data, pyaudio.paContinue
        return callback

    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

    def _get_volume(self, in_data):
        data = np.fromstring(in_data, dtype=np.int16)
        peak = np.max(np.abs(data)) * 2
        e = 0.0001
        volume = round(20 * math.log10((peak+e) / 2**16))
        return volume
    
    def _visualize_volume(self, volume):
        if volume < -50:
            bars = '[' + '-' * 50 + ']'
        elif volume >= -18 and volume < -5:
            bars = '[' + colored(' ' * (50 - 18), on_color='on_green') + colored(' ' * (18 + volume), on_color='on_yellow') + '-' * abs(volume) + ']'
        elif volume >= -5:
            bars = '[' + colored(' ' * (50 - 18), on_color='on_green') + colored(' ' * (18 - 5), on_color='on_yellow') + colored(' ' * (5+volume), on_color='on_red') + '-' * abs(volume) + ']'
        else:
            bars = '[' + colored(' ' * (50 + volume), on_color='on_green') + '-' * abs(volume) + ']'
        if volume < -150:
            num_str = '-∞'
        else:
            num_str = str(volume)
        print(bars + num_str, end='\r')