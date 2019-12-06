from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.rnn_model import RnnModel
from chord_recommendation.configs import MARKOV_ORDER, DROUPOUT_RATE, N_STEPS
from configs import *
from evaluation import *

# Initialize
markov = MarkovModel(MARKOV_ORDER)
markov.load(MARKOV_MODEL_PATH)
rnn = RnnModel()
rnn.load(RNN_MODEL_PATH, DROUPOUT_RATE)

# Parse the testing set, and get the cross entropy
ce = cross_entropy_markov(markov, TEST_PATH)
print('Markov order:', MARKOV_ORDER)
print('Cross-entropy for Markov chains:', ce)
ce = cross_entropy_rnn(rnn, TEST_PATH)
print('RNN steps:', N_STEPS)
print('Cross-entropy for RNN:', ce)

# Test the result
#progression_1 = [('C', 'maj'), ('A', 'min'), ('F', 'maj')]
#progression_2 = [('A', 'min'), ('F', 'maj'), ('C', 'maj')]
#progression_3 = [('C', 'maj'), ('G', 'maj'), ('A', 'min')]
#
#print('Markov prediction of the progression 1')
#print(markov.predict(progression_1)[0])
#
#print('Markov prediction of the progression 2')
#print(markov.predict(progression_2)[0])
#
#print('Markov prediction of the progression 3')
#print(markov.predict(progression_3)[0])
#
#print('RNN prediction of the progression 1')
#print(rnn.predict(progression_1)[0])
#
#print('RNN prediction of the progression 2')
#print(rnn.predict(progression_2)[0])
#
#print('RNN prediction of the progression 3')
#print(rnn.predict(progression_3)[0])