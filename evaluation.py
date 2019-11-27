from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.mcgill_parser import McGillParser
from chord_recommendation.configs import MARKOV_ORDER
from chord_recommendation.utils import *

def cross_entropy(markov: MarkovModel, path: str) -> float:
    parser = McGillParser()
    mean_cross_entropy = 0
    n_seqs = 0
    for chords in parser.parse_directory('mcgill-test'):
        if len(chords) < MARKOV_ORDER + 1:
            continue
        chord_ids = chords_to_ids(chords)
        onehot_mat = chords_to_onehot_mat(chords)
        ground_truths = onehot_mat[MARKOV_ORDER:]
        predictions = np.zeros((onehot_mat.shape[0]-MARKOV_ORDER,
                                onehot_mat.shape[1]))
        for i in range(len(chords) - MARKOV_ORDER):
            state = chord_ids[i: i+MARKOV_ORDER]
            predictions[i, :] = markov.predict_by_id(state)

        mean_cross_entropy += get_cross_entropy(predictions, ground_truths)
        n_seqs += 1
    mean_cross_entropy /= n_seqs
    return mean_cross_entropy