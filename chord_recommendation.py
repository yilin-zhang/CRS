from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.mcgill_parser import McGillParser
from chord_recommendation.utils import *
from chord_recommendation.configs import MARKOV_ORDER
from config import *

# Initialize
markov = MarkovModel(MARKOV_ORDER)
markov.load(MARKOV_MODEL_PATH)
parser = McGillParser()

# Parse the testing set, and get the cross entropy
mean_cross_entropy = 0
n_seqs = 0
for chords in parser.parse_directory('mcgill-test'):
    if len(chords) < MARKOV_ORDER + 1:
        continue
    chord_nums = chords_to_nums(chords)
    onehot_mat = chords_to_onehot_mat(chords)
    ground_truths = onehot_mat[MARKOV_ORDER:]
    predictions = np.zeros((onehot_mat.shape[0]-MARKOV_ORDER,
                            onehot_mat.shape[1]))
    for i in range(len(chords) - MARKOV_ORDER):
        state = chord_nums[i: i+MARKOV_ORDER]
        predictions[i, :] = markov.predict(state)

    mean_cross_entropy += get_cross_entropy(predictions, ground_truths)
    n_seqs += 1
mean_cross_entropy /= n_seqs
print('cross entropy:', mean_cross_entropy)

# Test the result
def predict_chords(progression):
    chord_seq = chords_to_nums(progression)
    prediction = markov.predict(chord_seq)
    order = np.argsort(prediction)[::-1].tolist()
    predicted_chords = list(map(num_to_chord, order))
    return predicted_chords

progression_1 = [('C', 'maj'), ('A', 'min'), ('F', 'maj')]
progression_2 = [('A', 'min'), ('F', 'maj'), ('C', 'maj')]
progression_3 = [('C', 'maj'), ('G', 'maj'), ('A', 'min')]

print('Prediction of the progression 1')
print(predict_chords(progression_1))

print('Prediction of the progression 2')
print(predict_chords(progression_2))

print('Prediction of the progression 3')
print(predict_chords(progression_3))