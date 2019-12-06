'''
Set up the program, create necessary files.
'''
import os
import sys
import shutil
import urllib3
import tarfile

from chord_recommendation.mcgill_parser import McGillParser
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel
from chord_recommendation.configs import *
from chord_recommendation.utils import chords_to_ids, transpose_chord_ids, gen_batch
from configs import *

def setup_dataset():
    # initialize
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)
    if not os.path.exists(MCGILL_PATH):
        url = 'https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1'
        file_path = DATASET_PATH + 'mcgill-billboard.tar.gz'
        http = urllib3.PoolManager()
        # download and unzip dataset
        with http.request('GET', url, preload_content=False) as r, open(file_path, 'wb') as out_file: 
            shutil.copyfileobj(r, out_file)
        tf = tarfile.open(file_path, 'r:gz')
        tf.extractall(DATASET_PATH)
        # split training and testing set
        os.mkdir(MCGILL_PATH+'train')
        os.mkdir(MCGILL_PATH+'test')
        for dir_name in os.listdir(MCGILL_PATH):
            if dir_name != 'train' and dir_name != 'test':
                if dir_name < '0903':
                    shutil.move(MCGILL_PATH+dir_name, MCGILL_PATH+'train/')
                else:
                    shutil.move(MCGILL_PATH+dir_name, MCGILL_PATH+'test/')

def setup_markov():
    setup_dataset()
    # Train the Markov model, save the model to `cache` directory
    parser = McGillParser()
    markov = MarkovModel(order=MARKOV_ORDER)
    with markov.train_batch():
        if not os.path.isdir(TRAIN_PATH):
            raise Exception("Training data folder does not exist. Please make sure folder exists and has the correct name")
        for chords in parser.parse_directory(TRAIN_PATH):
            if len(chords) < MARKOV_ORDER + 1:
                continue
            chord_ids = chords_to_ids(chords)
            chord_seqs = [chord_ids] + transpose_chord_ids(chord_ids)
            markov.feed_seqs(chord_seqs)

    markov.serialize(MARKOV_MODEL_PATH)

def setup_rnn():
    setup_dataset()
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

def clean_cache():
    shutil.rmtree(CACHE_PATH)

if __name__ == '__main__':
    arg = sys.argv[1]

    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    if arg == 'markov':
        setup_markov()
    elif arg == 'rnn':
        setup_rnn()
    elif arg == 'clean':
        clean_cache()
