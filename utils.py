import numpy as np
from configs import *

def chord_to_onehot(chord):
    ''' Convert a chord to a one-hot array. It only supports triads.
    Arg:
    - chord: A tuple, such as ('C', 'maj')
    Return:
    - onehot: A one-hot array. Order: [Cmaj, Cmin, C#maj, C#min, ...]
    '''
    root_note, chord_type = chord

    try:
        root_idx = CHORD_SEQ_1.index(root_note)
    except ValueError:
        root_idx = CHORD_SEQ_2.index(root_note)

    onehot = np.zeros(12*N_TYPES)

    for idx, type_name in enumerate(TYPE_SEQ):
        if chord_type == type_name:
            onehot[root_idx*N_TYPES + idx] = 1

    return onehot

def onehot_to_chord(onehot):
    ''' Convert a one-hot array to a chord. It only supports triads.
    Arg:
    - onehot: A one-hot array. Order: [Cmaj, Cmin, C#maj, C#min, ...]
    Return:
    - chord: A tuple, such as ('C', 'maj')
    '''
    idx = np.where(onehot == 1)[0].tolist()[0]
    root_idx = idx // N_TYPES
    type_idx = idx % N_TYPES
    chord = (CHORD_SEQ_1[root_idx], TYPE_SEQ[type_idx])
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
    ''' Convert a one-hot matrix to chords. It only supports triads.
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


