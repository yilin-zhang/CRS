from mcgill_parser import McGillParser
from markov_model import MarkovModel
from utils import *
from configs import *

# Initialize parser and markov model
parser = McGillParser()
markov = MarkovModel(order=MARKOV_ORDER)

# Parse dataset, and feed the data into the model
for chords in parser.parse_directory('McGill-Billboard'):
    if len(chords) < MARKOV_ORDER + 1:
        continue
    chord_nums = chords_to_nums(chords)
    chord_seqs = [chord_nums] + transpose_chord_nums(chord_nums)
    markov.feed_seqs(chord_seqs)

# Get the transition probability matrix
markov.normalize()

# Test the result
progression = [('C', 'maj'), ('A', 'min'), ('G', 'maj')]
chord_seq = chords_to_nums(progression)
prediction = markov.predict(chord_seq)
order = np.argsort(prediction)[::-1].tolist()

# Print the chords, from the most probable to the least
for num in order:
    print(num_to_chord(num))