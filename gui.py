import re
import keyboard
import sys
import termios
from termcolor import colored

from setup import setup_dataset, setup_markov, setup_rnn, clean_cache
from audio_stream.audio_stream import AudioStream
from audio_stream.configs import TEMP_AUDIO_NAME
from configs import CACHE_PATH, MARKOV_MODEL_PATH, RNN_MODEL_PATH

from chord_recommendation.configs import *
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel

from chord_recognition.chord_recog import chord_recognition
from chroma_chord_detection import chord_detection_improved

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
            command = input(self.prompt).strip()
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
                self._print_fisrt_level_help()
                print('')

    def _run_recording(self, model):
        chords = []
        stream = AudioStream(self.temp_audio_path)
        stream.clean()
        while True:
            # Record audio
            while True:
                command = input(self.rec_prompt).strip()
                # recording
                if command == 'r':
                    stream.start()
                    print('Recording... Press <space> to stop.')
                    break
                # revert
                elif command == 'v':
                    if len(chords) == 0:
                        print('Please record at least 1 chord.\n')
                        continue
                    chords.pop()
                    print('Chain reverted:')
                    self._print_chord_chain(chords)
                    print('')
                # recommend chords 
                elif command == 'm':
                    appended_chord = self._get_chord_from_command('m', command)
                    if len(chords) < self.minimum_input:
                        print('More chords required.\n')
                        continue
                    self._print_chord_recommend(chords, model)
                    print('')
                # exit
                elif command == 'exit':
                    return
                # fix the latest chord, (update chords[])
                elif re.search(r'^f ', command):
                    fixed_chord = self._get_chord_from_command('f', command)
                    if not fixed_chord:
                        print('Illegal chord name or type.\n')
                        continue
                    chords.pop()
                    chords.append(fixed_chord)
                    print('Chord fixed:')
                    self._print_chord_chain(chords)
                    print('')
                # append a chord manually
                elif re.search(r'^a ', command):
                    appended_chord = self._get_chord_from_command('a', command)
                    if not appended_chord:
                        print('Illegal chord name or type.\n')
                        continue
                    chords.append(appended_chord)
                    print('Chord appended:')
                    self._print_chord_chain(chords)
                    print('')
                # print help
                else:
                    self._print_second_level_help()
                    print('')
                    continue
            
            # Recording, waiting for stop
            keyboard.wait('space')
            stream.stop()
            print(' ' * 56)
            termios.tcflush(sys.stdin, termios.TCIFLUSH) # flush input buffer
                
            # Chord recognition
            #chord_sequence = chord_recognition(self.temp_audio_path)
            chord = chord_detection_improved(self.temp_audio_path)
            chords.append(chord)
            #chords += chord_sequence
            stream.clean()
            # print chord chain
            self._print_chord_chain(chords)
            print('')
            # Chord recommendation
            if len(chords) >= self.minimum_input:
                self._print_chord_recommend(chords, model) 
                print('')

    def _print_fisrt_level_help(self):
        print("Command")
        print("  run rnn       Use RNN for chord recommendation.")
        print("  run markov    Use Markov chains for chord recommendation.")
        print("  setup rnn     Create RNN model.")
        print("  setup makrov  Create Markov model.")
        print("  clean         Clean cache directory, remove model files.")
        print("  exit          Exit the program.")

    def _print_second_level_help(self):
        print("Command")
        print("  r             Record.")
        print("  v             Revert the chord chain.")
        print("  m             Recommend chords based on the chord chain.")
        print("  a             Manually append a chord to the chord chain.")
        print("  f <new_chord> Fix the latest recognized chord.")
        print("  exit          Exit the recommending mode.")

    def _format_chord(self, chord):
        if chord[1] == 'min':
            chord_symbol = chord[0].lower()
        else:
            chord_symbol = chord[0]
        return chord_symbol
    
    def _print_chord_chain(self, chords):
        for ind, c in enumerate(chords):
            chord_symbol = self._format_chord(c)
            if (len(chords) - ind) <= self.minimum_input:
                print(colored(chord_symbol, color='green', attrs=['bold']), end='')
            else:
                print(chord_symbol, end='')
            if ind != len(chords) - 1:
                print(' -> ', end='')
            else:
                print('')
    
    def _print_chord_recommend(self, chords, model):
        # Prediction/Recommendation
        prediction = [None, None, None]
        if model == 'markov':
            prediction = self.markov.predict(chords)[0]
        elif model == 'rnn':
            prediction = self.rnn.predict(chords)[0]
        print('Recommendation for next chord:')
        for i in range(3):
            if i == 2:
                print(self._format_chord(prediction[i]))
            else:
                print(self._format_chord(prediction[i]), end=', ')
    
    def _get_chord_from_command(self, cmd_name, command):
        cmd_match = re.search(r'^' + cmd_name + r' ([a-g]|[A-G])(#?)', command)
        if cmd_match:
            if cmd_match.group(1).isupper():
                matched_chord = (
                    cmd_match.group(1)+cmd_match.group(2),
                    'maj'
                )
            else:
                matched_chord = (
                    cmd_match.group(1).upper()+cmd_match.group(2),
                    'min'
                )
            return matched_chord
        else:
            return None
