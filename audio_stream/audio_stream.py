from audio_stream.recorder import Recorder
import time
import os

# TODO: Specify the audio device, add volume meter
class AudioStream():
    def __init__(self, temp_audio_path):
        self.save_path = temp_audio_path

    def start(self):
        self.rec = Recorder().open(self.save_path)
        self.rec.start_recording()

    def stop(self):
        self.rec.stop_recording()
        self.rec.close()

    def clean(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
