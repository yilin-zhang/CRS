'''
Utility functions.
'''
from typing import *
import numpy as np
from chord_recommendation.configs import *
from chord_recommendation.mcgill_parser import McGillParser

# Chord representation conversions
def chord_to_id(chord: Tuple[str, str]) -> int:
    ''' Convert a chord to an integer number.
    Arg:
    - chord: A tuple, such as ('C', 'maj')
    Return:
    - chord_id: A number, which represent a chord
    '''
    root_note, chord_type = chord

    try:
        root_idx = CHORD_SEQ_1.index(root_note)
    except ValueError:
        root_idx = CHORD_SEQ_2.index(root_note)

    chord_id = 0

    for idx, type_name in enumerate(TYPE_SEQ):
        if chord_type == type_name:
            chord_id = root_idx * N_TYPES + idx
    
    return chord_id

def id_to_chord(chord_id: int) -> Tuple[str, str]:
    ''' Convert an integer number to a chord.
    Arg:
    - chord_id: A number, which represent a chord
    Return:
    - chord: A tuple, such as ('C', 'maj')
    '''
    root_idx = chord_id // N_TYPES
    type_idx = chord_id % N_TYPES
    chord = (CHORD_SEQ_1[root_idx], TYPE_SEQ[type_idx])
    return chord

def chords_to_ids(chords: List[Tuple[str, str]]) -> List[int]:
    ''' Convert a chord sequence to a number sequence
    Arg:
    - chords: A sequence of chords
    Return:
    - ids: A sequence of chord numbers
    '''
    ids = list(map(chord_to_id, chords))
    return ids

def ids_to_onehot_mat(ids: List[int]) -> np.ndarray:
    onehot_mat = np.zeros((len(ids), N_TYPES*12))
    for i in range(len(ids)):
        onehot = np.zeros(N_TYPES*12)
        onehot[ids[i]] = 1
        onehot_mat[i] = onehot
    return onehot_mat

def chord_to_onehot(chord: Tuple[str, str]) -> np.ndarray:
    ''' Convert a chord to a one-hot array.
    Arg:
    - chord: A tuple, such as ('C', 'maj')
    Return:
    - onehot: A one-hot array. Order: [Cmaj, Cmin, C#maj, C#min, ...]
    '''
    onehot = np.zeros(12*N_TYPES)
    onehot[chord_to_id(chord)] = 1
    return onehot

def onehot_to_chord(onehot: np.ndarray) -> Tuple[str, str]:
    ''' Convert a one-hot array to a chord.
    Arg:
    - onehot: A one-hot array. Order: [Cmaj, Cmin, C#maj, C#min, ...]
    Return:
    - chord: A tuple, such as ('C', 'maj')
    '''
    idx = np.where(onehot == 1)[0].tolist()[0]
    chord = id_to_chord(idx)
    return chord

def chords_to_onehot_mat(chords: List[Tuple[str, str]]) -> np.ndarray:
    ''' Convert chords to a series of one-hot arrays (a matrix)
    Arg:
    - chords: A sequence of chords
    Return:
    - onehot_mat: A matrix, each row is a one-hot array.
    '''
    onehot_mat = np.array(list(map(chord_to_onehot, chords)))
    return onehot_mat

def onehot_mat_to_chords(onehot_mat: np.ndarray) -> List[Tuple[str, str]]:
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

# Chord transposition
def transpose_chord_ids(chord_ids: List[int]) -> List[List[int]]:
    ''' Transpose a seires of chord numbers
    Arg:
    - chord_ids: A list of chord numbers
    Return
    - transposed_list: A list of transposed chord number lists
      [   ] <- transposed_list
      [ ]   <- 11 transposed chord_ids
    '''
    transposed_list = []
    for i in range(1, 12):
        transposed_list.append([(x+N_TYPES*i) % (N_TYPES*12) for x in chord_ids])
    return transposed_list

def transpose_onehot_mat(onehot_mat: np.ndarray) -> List[np.ndarray]:
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

def gen_batch(path: str, n_steps: int, batch_size: int, checking: bool = False) -> Iterator[Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]]:
    parser = McGillParser()
    while True:
        X_batch = []
        output_batch = []
        for chords in parser.parse_directory(path):
            if len(chords) < N_STEPS + 1:
                continue
            chord_ids = chords_to_ids(chords)
            onehot_mat = chords_to_onehot_mat(chords)
            transposed_onehot_mats = [onehot_mat] + transpose_onehot_mat(onehot_mat)
            for i in range(len(chords) - N_STEPS):
                for j in range(12):
                    X_batch.append(transposed_onehot_mats[j][i:i+N_STEPS])
                    output_batch.append(transposed_onehot_mats[j][i+1:i+N_STEPS+1])
                    if len(X_batch) == batch_size:
                        yield (
                            np.array(X_batch),
                            {
                                'output': np.array(output_batch)
                            }
                        )
                        X_batch = []
                        output_batch = []
        if checking:
            break