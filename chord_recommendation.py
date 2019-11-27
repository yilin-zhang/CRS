from chord_recommendation.markov_model import MarkovModel
from chord_recommendation.configs import MARKOV_ORDER
from config import *
from evaluation import cross_entropy

# Initialize
markov = MarkovModel(MARKOV_ORDER)
markov.load(MARKOV_MODEL_PATH)

# Parse the testing set, and get the cross entropy
ce = cross_entropy(markov, TEST_PATH)
print(ce)

# Test the result
progression_1 = [('C', 'maj'), ('A', 'min'), ('F', 'maj')]
progression_2 = [('A', 'min'), ('F', 'maj'), ('C', 'maj')]
progression_3 = [('C', 'maj'), ('G', 'maj'), ('A', 'min')]

print('Prediction of the progression 1')
print(markov.predict(progression_1))

print('Prediction of the progression 2')
print(markov.predict(progression_2))

print('Prediction of the progression 3')
print(markov.predict(progression_3))