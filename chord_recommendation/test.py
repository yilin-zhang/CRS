import unittest
import numpy as np
from chord_recommendation.utils import *

class TestUtils(unittest.TestCase):
    def test_chord_conversion(self):
        chords = [('C', 'maj'), ('D', 'min')]
        self.assertEqual(chords,
            onehot_mat_to_chords(chords_to_onehot_mat(chords)))

if __name__ == '__main__':
    unittest.main()
