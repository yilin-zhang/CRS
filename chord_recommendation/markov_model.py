from chord_recommendation.configs import MARKOV_ORDER, N_TYPES
import numpy as np

class MarkovModel():
    def __init__(self, order=2):
        self.order = order
        self.freq_mat = np.ones((12*N_TYPES,)*(self.order+1),
                                 dtype=np.uint)

    def train(self, seqs):
        self.feed_seqs(seqs)
        self._ormalize()

    def predict(self, seq):
        seq_len = len(seq)
        if seq_len < self.order:
            raise ValueError(
                'The length of seqs should not be less than order.'
            )
        idx = tuple(seq[-(self.order):])
        return self.trans_prob_mat[idx]

    def clean(self):
        '''Clean up the model.'''
        self.freq_mat = np.zeros((12*N_TYPES,)*(self.order+1),
                                 dtype=np.uint)

    def feed_seq(self, seq):
        '''Feed a sequence into the model.
        Arg:
        - seq: A sequence, which contains the information of state transitions
        '''
        seq_len = len(seq)
        if seq_len < self.order + 1:
            raise ValueError(
                'The length of seqs should not be less than order+1.'
            )
        for i in range(len(seq) - self.order):
            idx = tuple(seq[i: i+self.order+1])
            self.freq_mat[idx] += 1

    def feed_seqs(self, seqs):
        '''Feed several sequences into the model.
        Arg:
        - seq: A list, which contains multiple sequences
        '''
        for seq in seqs:
            self.feed_seq(seq)
    
    def normalize(self):
        trans_prob_mat = np.zeros((12*N_TYPES,)*(self.order+1))
        for i in range(12*N_TYPES):
            trans_prob_mat[:, :, i] = self.freq_mat[:, :, i] / np.sum(self.freq_mat[:, :, i])
        self.trans_prob_mat = trans_prob_mat
    