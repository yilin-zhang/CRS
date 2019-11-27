from chord_recommendation.configs import MARKOV_ORDER, N_TYPES
from contextlib import contextmanager
import warnings
import pickle
import numpy as np

class MarkovModel():
    def __init__(self, order=2):
        self._order = order
        self._freq_mat = np.ones((12*N_TYPES,)*(self._order+1),
                                 dtype=np.uint)
        self._is_trained = False

    def train(self, seqs):
        self._feed_seqs(seqs)
        self._normalize()
        self._is_trained = True

    @contextmanager
    def train_batch(self):
        ''' Using with key word to automatically normalize it
        after feeding data
        '''
        yield
        self._normalize()
        self._is_trained = True

    def predict(self, seq):
        seq_len = len(seq)
        if seq_len < self._order:
            raise ValueError(
                'The length of seqs should not be less than order.'
            )
        idx = tuple(seq[-(self._order):])
        return self.trans_prob_mat[idx]

    def clean(self):
        '''Clean up the model.'''
        self._freq_mat = np.zeros((12*N_TYPES,)*(self._order+1),
                                 dtype=np.uint)
        self._is_trained = False

    def feed_seq(self, seq):
        '''Feed a sequence into the model.
        Arg:
        - seq: A sequence, which contains the information of state transitions
        '''
        seq_len = len(seq)
        if seq_len < self._order + 1:
            raise ValueError(
                'The length of seqs should not be less than order+1.'
            )
        for i in range(len(seq) - self._order):
            idx = tuple(seq[i: i+self._order+1])
            self._freq_mat[idx] += 1

    def feed_seqs(self, seqs):
        '''Feed several sequences into the model.
        Arg:
        - seq: A list, which contains multiple sequences
        '''
        for seq in seqs:
            self.feed_seq(seq)
    
    def _normalize(self):
        trans_prob_mat = np.zeros((12*N_TYPES,)*(self._order+1))
        for i in range(12*N_TYPES):
            trans_prob_mat[:, :, i] = self._freq_mat[:, :, i] / np.sum(self._freq_mat[:, :, i])
        self.trans_prob_mat = trans_prob_mat
    
    def serialize(self, path):
        ''' Serialize the transition probability matrix.
        Arg:
        - path: The path to the serialized file
        '''
        if not self._is_trained:
            raise Exception('The Markov model has not been trained.')
        with open(path, 'wb') as f:
            pickle.dump(self.trans_prob_mat, f)

    def load(self, path):
        ''' Load the transition probability matrix.
        Arg:
        - path: The path to the serialized file
        '''
        if self._is_trained:
            warnings.warn('The model is trained. Loading a model will overwrite the current one.')
        with open(path, 'rb') as f:
            trans_prob_mat = pickle.load(f)
        new_order = len(trans_prob_mat.shape) - 1
        if new_order != self._order:
            warnings.warn('The new order is: ' + str(new_order))
            self._order = new_order
        self.trans_prob_mat = trans_prob_mat
        self._is_trained = True

    