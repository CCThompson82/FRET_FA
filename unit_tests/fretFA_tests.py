
import os
import sys
ROOT_DIR = os.environ['PWD']
sys.path.append(ROOT_DIR)

import unittest
import numpy as np

from fretFA import Experiment

# placeholders
no_acceptor_path = 'Experiment_test/Control mTFP1'
no_donor_path = 'Experiment_test/Control Venus'
no_cell_path = 'Experiment_test/Background'
sample_path_list = ['Experiment_test/Uninfected/Vinculin-TL', 'Experiment_test/Ctr Infected/Vinculin-TL', 'Experiment_test/Ctr Infected/Vinculin-TS']

path_bundle = [no_acceptor_path, no_donor_path, no_cell_path, sample_path_list]

test_experiment = Experiment(path_bundle)

class TestFretFA(unittest.TestCase) :
    """
    Suite of tests to assert function of the FretFA module for analysis of
    focal adhesion modification using a FRET based reporting system.
    """
    def test_init(self) :
        """
        Ensures Experiment object is initiated by checking if a basic instance
        variable is defined properly.
        """
        self.assertEqual(test_experiment.no_acceptor_path, no_acceptor_path)

    def test_bleedthrough_no_acceptor(self):
        """
        Runs calculate_bleedthrough on test data and checks the expected
        regression coefficient is returned.
        """
        self.assertAlmostEqual(test_experiment.calculate_bleedthrough('no_acceptor_control', bins = 16)[0], 0.9347425)

    def test_bleedthrough_co_donor(self):
            """
            Runs calculate_bleedthrough on test data and checks the expected
            regression coefficient is returned.
            """ 
        self.assertAlmostEqual(test_experiment.calculate_bleedthrough('no_donor_control', bins = 16)[0], 0.4232927)




if __name__ == '__main__':
    unittest.main()
