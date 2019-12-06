'''
All the configurations.

To support more chord types, add a constant `SYM_XXX`, and add it to the
`TYPE_SEQ`.
'''

# All possible chord names
CHORD_SEQ_1 = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORD_SEQ_2 = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Chord types
SYM_MAJ = 'maj'
SYM_MIN = 'min'
TYPE_SEQ = [SYM_MAJ, SYM_MIN]
N_TYPES = len(TYPE_SEQ)

# Markov chains
MARKOV_ORDER = 5

# RNN
N_STEPS = 5
N_INPUT = N_TYPES * 12
N_NEURONS = 64
DROUPOUT_RATE = 0.3
BATCH_SIZE = 5
N_EPOCHS = 50