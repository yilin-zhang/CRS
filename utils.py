'''
Utility functions.
'''

import numpy as np
from configs import *

def chord_to_num(chord):
    ''' Convert a chord to an integer number.
    Arg:
    - chord: A tuple, such as ('C', 'maj')
    Return:
    - chord_num: A number, which represent a chord
    '''
    root_note, chord_type = chord

    try:
        root_idx = CHORD_SEQ_1.index(root_note)
    except ValueError:
        root_idx = CHORD_SEQ_2.index(root_note)

    chord_num = None

    for idx, type_name in enumerate(TYPE_SEQ):
        if chord_type == type_name:
            chord_num = root_idx * N_TYPES + idx
    
    return chord_num

def num_to_chord(chord_num):
    ''' Convert a chord to an integer number.
    Arg:
    - chord_num: A number, which represent a chord
    Return:
    - chord: A tuple, such as ('C', 'maj')
    '''
    root_idx = chord_num // N_TYPES
    type_idx = chord_num % N_TYPES
    chord = (CHORD_SEQ_1[root_idx], TYPE_SEQ[type_idx])
    return chord

def chord_to_onehot(chord):
    ''' Convert a chord to a one-hot array.
    Arg:
    - chord: A tuple, such as ('C', 'maj')
    Return:
    - onehot: A one-hot array. Order: [Cmaj, Cmin, C#maj, C#min, ...]
    '''
    onehot = np.zeros(12*N_TYPES)
    onehot[chord_to_num(chord)] = 1
    return onehot

def onehot_to_chord(onehot):
    ''' Convert a one-hot array to a chord.
    Arg:
    - onehot: A one-hot array. Order: [Cmaj, Cmin, C#maj, C#min, ...]
    Return:
    - chord: A tuple, such as ('C', 'maj')
    '''
    idx = np.where(onehot == 1)[0].tolist()[0]
    chord = num_to_chord(idx)
    return chord

def chords_to_onehot_mat(chords):
    ''' Convert chords to a series of one-hot arrays (a matrix)
    Arg:
    - chords: A sequence of chords
    Return:
    - onehot_mat: A matrix, each row is a one-hot array.
    '''
    onehot_mat = np.array(list(map(chord_to_onehot, chords)))
    return onehot_mat

def onehot_mat_to_chords(onehot_mat):
    ''' Convert a one-hot matrix to chords.
    Arg:
    - onehot_mat: A one-hot matrix
    Return:
    - chords: A sequence of chords
    '''
    chords = []
    for i in range(onehot_mat.shape[0]):
        chords.append(onehot_to_chord(onehot_mat[i, :]))
    return chords

def transpose_onehot_mat(onehot_mat):
    ''' Transpose a one-hot matrix (a chord sequence).
    Arg:
    - onehot_mat: A one-hot matrix
    Return:
    - transposed_mats: A list that contains all the transposed matrices.
    '''
    transposed_mats = []
    for i in range(1, 12):
        transposed_mats.append(np.roll(onehot_mat, N_TYPES*i, axis=1))
    return transposed_mats
