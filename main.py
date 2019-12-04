import time

from audio_stream.audio_stream import AudioStream
from audio_stream.configs import TEMP_AUDIO_NAME
from configs import CACHE_PATH, MARKOV_MODEL_PATH, RNN_MODEL_PATH

from chord_recommendation.configs import *
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel

# TODO: Put the function into the module
from chroma_chord_detection import chord_detection

# Depends on using Markov or RNN
minimum_input = MARKOV_ORDER
#minimum_input = N_STEPS

temp_audio_path = CACHE_PATH + TEMP_AUDIO_NAME

chords = []

markov = MarkovModel(MARKOV_ORDER)
markov.load(MARKOV_MODEL_PATH)
rnn = RnnModel()
rnn.load(RNN_MODEL_PATH, DROUPOUT_RATE)

stream = AudioStream(temp_audio_path)
stream.clean()
while True:
    # Record audio
    stream.start()
    print('Start recording...')
    time.sleep(3)
    stream.stop()
    print('Stop recording...')

    # Chord recognition
    chord = chord_detection(temp_audio_path)
    print('You just played:', chord)
    chords.append(chord)
    stream.clean()

    if len(chords) == minimum_input:
        prediction = rnn.predict(chords)
        print('Recommend:', [prediction[0][0], prediction[0][1], prediction[0][2]])
        chords = chords[1:]


