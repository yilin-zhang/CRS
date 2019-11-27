'''
Set up the program, create necessary files.
'''

from chord_recommendation.mcgill_parser import McGillParser
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.configs import MARKOV_ORDER
from chord_recommendation.utils import chords_to_nums, transpose_chord_nums
from config import MARKOV_MODEL_PATH

# Train the Markov model, save the model to `cache` directory
parser = McGillParser()
markov = MarkovModel(order=MARKOV_ORDER)
with markov.train_batch():
    for chords in parser.parse_directory('mcgill-train'):
        if len(chords) < MARKOV_ORDER + 1:
            continue
        chord_nums = chords_to_nums(chords)
        chord_seqs = [chord_nums] + transpose_chord_nums(chord_nums)
        markov.feed_seqs(chord_seqs)
markov.serialize(MARKOV_MODEL_PATH)