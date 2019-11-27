'''
Set up the program, create necessary files.
'''
import os
import sys

from chord_recommendation.mcgill_parser import McGillParser
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel
from chord_recommendation.configs import *
from chord_recommendation.utils import chords_to_ids, transpose_chord_ids, gen_batch
from config import *

arg = sys.argv[1]

if arg == 'markov':
    # Train the Markov model, save the model to `cache` directory
    parser = McGillParser()
    markov = MarkovModel(order=MARKOV_ORDER)
    with markov.train_batch():
        for chords in parser.parse_directory(TRAIN_PATH):
            if len(chords) < MARKOV_ORDER + 1:
                continue
            chord_ids = chords_to_ids(chords)
            chord_seqs = [chord_ids] + transpose_chord_ids(chord_ids)
            markov.feed_seqs(chord_seqs)

    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    markov.serialize(MARKOV_MODEL_PATH)

elif arg == 'rnn':
    # Train the RNN model, save the model to `cache` directory
    steps_per_epoch = 0
    for _ in gen_batch(TRAIN_PATH, N_STEPS, BATCH_SIZE, True):
        steps_per_epoch += 1

    validation_steps = 0
    for _ in gen_batch(TEST_PATH, N_STEPS, BATCH_SIZE, True):
        validation_steps += 1

    rnn_model = RnnModel()
    rnn_model.construct(DROUPOUT_RATE)
    rnn_model.compile()
    rnn_model.fit(
        TRAIN_PATH,
        TEST_PATH,
        BATCH_SIZE,
        steps_per_epoch,
        validation_steps,
        N_EPOCHS,
        CACHE_PATH,
        CACHE_PATH + 'log/'
    )