import numpy as np
from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel
from chord_recommendation.mcgill_parser import McGillParser
from chord_recommendation.configs import *
from chord_recommendation.utils import *

def get_cross_entropy(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    ''' Calculate cross entropy
    Args:
    - predictions: A matrix, each row represents a sample.
    - groundtruths: A matrix, each row is a one-hot array.
    Return:
    - cross_entropy
    '''
    if len(predictions.shape) == 1:
        n_samples = 1
    else:
        n_samples = predictions.shape[0]
    cross_entropy_arr = np.zeros(n_samples)
    for i in range(n_samples):
        cross_entropy_arr[i] = - np.sum(ground_truths[i] * np.log(predictions[i]))
    cross_entropy = np.mean(cross_entropy_arr)
    return cross_entropy

def cross_entropy_markov(markov: MarkovModel, path: str) -> float:
    parser = McGillParser()
    mean_cross_entropy = 0
    n_seqs = 0
    for chords in parser.parse_directory(path):
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

def cross_entropy_rnn(rnn: RnnModel, path: str) -> float:
    parser = McGillParser()
    mean_cross_entropy = 0
    n_seqs = 0
    for chords in parser.parse_directory(path):
        if len(chords) < N_STEPS + 1:
            continue
        chord_ids = chords_to_ids(chords)
        onehot_mat = chords_to_onehot_mat(chords)
        ground_truths = onehot_mat[N_STEPS:]
        predictions = np.zeros((onehot_mat.shape[0]-N_STEPS,
                                onehot_mat.shape[1]))
        for i in range(len(chords) - N_STEPS):
            state = np.zeros((len(chords) - N_STEPS, 12*N_TYPES))
            state = ids_to_onehot_mat(chord_ids[i: i+N_STEPS])
            predictions[i, :] = rnn.predict_onehot_batch(np.array([state]))[0]

        mean_cross_entropy += get_cross_entropy(predictions, ground_truths)
        n_seqs += 1
    mean_cross_entropy /= n_seqs
    return mean_cross_entropy