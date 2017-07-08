
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

    def test_background_adj_calc(self):
        test_experiment.background_adjustments_calc()
        self.assertAlmostEqual(test_experiment.mean_channel_background[0], 4.4231016)

    def test_remove_maxouts(self) :
        """
        Artificial test for function
        """
        arr = np.array([[1,1],[0, (2**16)-1]])
        adj_arr = test_experiment.remove_maxouts(arr)
        self.assertEqual(np.sum(adj_arr), 2)

    def test_subtract_background(self):
        """
        Check for subtract_background fn using test_data and artificial img arr
        """
        if not hasattr(test_experiment, 'mean_channel_background') :
            test_experiment.background_adjustments_calc()
        else :
            pass
        arr = np.ones([2,2,3])
        adj_arr = test_experiment.subtract_background(arr)
        self.assertLess(np.sum(adj_arr), np.sum(arr))

    def test_threshold_filter(self) :
        """
        Check that pixels below the threshold get zero'd with
        `self.threshold_filter`.
        """
        arr = np.ones([2,2,3])
        arr[0,0,:] = 100
        adj_arr = test_experiment.threshold_filter(arr)
        self.assertLess(np.min(adj_arr), np.min(arr))

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


    def test_cFRET(self):
        """
        Test the master function for background removal and image adjustments.
        """

        test_experiment.background_adjustments_calc()
        test_experiment.define_bt()

        test_img = np.random.randint(0,255, 12).reshape([2,2,3])

        ret_img = test_experiment.cFRET(test_img)
        self.assertGreater(np.sum(test_img), np.sum(ret_img))



if __name__ == '__main__':
    unittest.main()
