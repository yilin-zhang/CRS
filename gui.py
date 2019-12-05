import math
import time
import sys
from termcolor import colored, cprint

from setup import setup_markov, setup_rnn, clean_cache
from audio_stream.audio_stream import AudioStream
from audio_stream.configs import TEMP_AUDIO_NAME
from configs import CACHE_PATH, MARKOV_MODEL_PATH, RNN_MODEL_PATH

from chord_recommendation.configs import *
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel

# TODO: Put the function into the module
from chroma_chord_detection import chord_detection


class CliGui():
    def __init__(self):
        self.prompt = '> '
        self.rec_prompt = colored('> ', color='red')
        self.temp_audio_path = CACHE_PATH + TEMP_AUDIO_NAME

    def run(self):
        ''' high level command '''
        crs_logo = colored("   __________  _____\n  / ____/ __ \\/ ___/\n / /   / /_/ /\\__ \\ \n/ /___/ _, _/___/ / \n\\____/_/ |_|/____/  ", color='blue', attrs=['bold'])
        print(crs_logo)
        crs_description = colored('C', color='blue', attrs=['bold']) + \
                          'hord ' + \
                          colored('R', color='blue', attrs=['bold']) + \
                          'ecommendation ' + \
                          colored('S', color='blue', attrs=['bold']) + \
                          'ystem'
        print(crs_description)
        # The main loop
        while True:
            command = input(self.prompt)
            if command == 'run rnn':
                self.minimum_input = N_STEPS
                self.rnn = RnnModel()
                self.rnn.load(RNN_MODEL_PATH, DROUPOUT_RATE)
                self._run_recording('rnn')
            elif command == 'run markov':
                self.minimum_input = MARKOV_ORDER
                self.markov = MarkovModel(MARKOV_ORDER)
                self.markov.load(MARKOV_MODEL_PATH)
                self._run_recording('markov')
            elif command == 'clean':
                clean_cache()
            elif command == 'setup rnn':
                setup_rnn()
            elif command == 'setup markov':
                setup_markov()
            elif command == 'exit':
                print('Goodbye!')
                exit()
            else:
                print("Command")
                print("  run rnn       Use RNN for chord recommendation.")
                print("  run markov    Use Markov chains for chord recommendation.")
                print("  setup rnn     Create RNN model.")
                print("  setup makrov  Create Markov model.")
                print("  clean         Clean cache directory, remove model files.")
                print("  exit          Exit the program.")

    def _run_recording(self, model):
        chords = []
        stream = AudioStream(self.temp_audio_path)
        stream.clean()
        while True:
            # Record audio
            while True:
                command = input(self.rec_prompt)
                if command == 's':
                    stream.start()
                    print('Recording...')
                    break
                elif command == 'exit':
                    return
                else:
                    continue
            # Wait for stop
            while True:
                command = input(self.rec_prompt)
                if command == 's':
                    stream.stop()
                    break
                else:
                    continue
                
            # Chord recognition
            chord = chord_detection(self.temp_audio_path)
            chords.append(chord)
            stream.clean()
            for ind, c in enumerate(chords):
                chord_symbol = self._format_chord(c)
                if (len(chords) - ind) <= self.minimum_input:
                    print(colored(chord_symbol, color='green', attrs=['bold']), end='')
                else:
                    print(chord_symbol, end='')
                if ind != len(chords) - 1:
                    print(' -> ', end='')
                else:
                    print('\n')
                
            # Prediction/Recommendation
            if len(chords) >= self.minimum_input:
                prediction = [None, None, None]
                if model == 'markov':
                    prediction = self.markov.predict(chords)[0]
                elif model == 'rnn':
                    prediction = self.rnn.predict(chords)[0]
                print('Maybe you want to try:')
                for i in range(3):
                    if i == 2:
                        print(self._format_chord(prediction[i]))
                    else:
                        print(self._format_chord(prediction[i]), end=', ')
                print('')
                #chords = chords[1:]
    
    def _format_chord(self, chord):
        if chord[1] == 'min':
            chord_symbol = chord[0].lower()
        else:
            chord_symbol = chord[0]
        return chord_symbol

    def _show_reording(self):
        print(self._get_volume_meter())
        pass

    def _update_volume(self, volume):
        self.volume = volume

    def _get_volume_meter(self):
        if self.volume < -50:
            bars = '[' + '-' * 50 + ']'
        elif self.volume >= -18 and self.volume < -5:
            bars = '[' + colored(' ' * (50 - 18), on_color='on_green') + colored(' ' * (18 + self.volume), on_color='on_yellow') + '-' * abs(self.volume) + ']'
        elif self.volume >= -5:
            bars = '[' + colored(' ' * (50 - 18), on_color='on_green') + colored(' ' * (18 - 5), on_color='on_yellow') + colored(' ' * (5+self.volume), on_color='on_red') + '-' * abs(self.volume) + ']'
        else:
            bars = '[' + colored(' ' * (50 + self.volume), on_color='on_green') + '-' * abs(self.volume) + ']'
        if self.volume < -150:
            num_str = '-∞'
        else:
            num_str = str(self.volume)
        return bars + ' ' + num_str
