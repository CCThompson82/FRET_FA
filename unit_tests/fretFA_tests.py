
import os
import sys
ROOT_DIR = os.environ['PWD']
sys.path.append(ROOT_DIR)

import unittest
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from fretFA import Experiment, SampleImage
from utils import *


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
        no_acceptor_path = 'Experiment_test/Control mTFP1'
        self.assertEqual(test_experiment.no_acceptor_path, no_acceptor_path)

    def test_dict_setup(self) :
        """
        Checks dictionary that is initiated from Experiment_test data is
        correct.
        """
        key = 'Experiment_test/Uninfected/Vinculin-TL'
        value_0 = '2016-03-16 FRET.lif - Uninfected 8h vinculin TL 1 low venus.tif'
        self.assertTrue(test_experiment.samples_dict[key][0] == value_0 )

    def test_background_adj_calc(self):
        """
        `Experiment.background_adjustments_calc`
        """
        expected_background = 4.4231016

        test_experiment.background_adjustments_calc()
        self.assertAlmostEqual(test_experiment.mean_channel_background[0],
            expected_background)

    def test_remove_maxouts(self) :
        """
        Theoretical fn test for `Experiment.remove_maxouts`
        """
        exp_result = 2
        arr = np.array([[1,1],[0, (2**16)-1]])
        adj_arr = remove_maxouts(arr)
        self.assertEqual(np.sum(adj_arr), exp_result)

    def test_subtract_background(self):
        """
        Theoretical `Experiment.subtract_background` test.
        """
        arr = np.ones([2,2,3])
        adj_arr = test_img.subtract_background()
        self.assertLess(np.sum(adj_arr), np.sum(test_img.img))

    def test_threshold_filter(self) :
        """
        Check that pixels below the threshold get zero'd with
        `Experiment.threshold_filter`.
        """
        arr = np.ones([2,2,3])
        arr[0,0,:] = 100
        adj_arr = test_img.threshold_filter()
        self.assertLess(np.min(adj_arr), np.min(arr))

    def test_bleedthrough_no_acceptor(self):
        """
        Runs calculate_bleedthrough on test data and checks the expected
        regression coefficient is returned.
        """
        exp_result = 0.9347425
        self.assertAlmostEqual(
            test_experiment.calculate_bleedthrough(
                'no_acceptor_control', bins = 16)[0],
                exp_result)

    def test_bleedthrough_co_donor(self):
        """
        Runs calculate_bleedthrough on test data and checks the expected
        regression coefficient is returned.
        """
        exp_result = 0.4232927
        self.assertAlmostEqual(
            test_experiment.calculate_bleedthrough(
                'no_donor_control', bins = 16)[0],
                exp_result)


    def show_image(self) :

        test_experiment.background_adjustments_calc()
        test_experiment.define_bt()

        key = 'Experiment_test/Uninfected/Vinculin-TL'
        test_img = io.imread(os.path.join(key, test_experiment.samples_dict[key][0]))

        ret_img = test_experiment.cFRET(test_img)
        f_img = test_experiment.boxfilter(ret_img)

        fig, axarr = plt.subplots(1,2, figsize = (10,5))
        axarr[0].imshow(ret_img)
        axarr[1].imshow(test_experiment.threshold_filter_gs(f_img))
        plt.show()

    def test_waterfall(self):
        """
        Test the waterfall segmentation function.
        """

        self.assertEqual(
            len(
                test_img.master_dict
               ), 91)

    def show_mask(self) :
        test_experiment.background_adjustments_calc()
        test_experiment.define_bt()

        key = 'Experiment_test/Uninfected/Vinculin-TL'
        test_img = io.imread(os.path.join(key, test_experiment.samples_dict[key][0]))

        #ret_img = test_experiment.cFRET(test_img)
        f_img = test_experiment.boxfilter(test_img[:,:,1])
        master_dict = test_experiment.waterfall_segmentation(
            f_img,
            I_threshold = 0.5,
            merger_threshold = 12,
            min_pix_area=8,
            verbose = True)

        mask_img = test_experiment.generate_mask(master_dict, f_img)

        fig, axarr = plt.subplots(2,2, figsize = (10,10))
        axarr[0,0].imshow(test_img[:,:,1])
        axarr[0,1].imshow(test_img[:,:,0])
        axarr[1,0].imshow(f_img)
        axarr[1,1].imshow(mask_img, cmap = 'nipy_spectral')
        plt.show()



if __name__ == '__main__':
    # placeholders
    no_acceptor_path = 'Experiment_test/Control mTFP1'
    no_donor_path = 'Experiment_test/Control Venus'
    no_cell_path = 'Experiment_test/Background'
    sample_path_list = ['Experiment_test/Uninfected/Vinculin-TL', 'Experiment_test/Ctr Infected/Vinculin-TL', 'Experiment_test/Ctr Infected/Vinculin-TS']
    data_root = 'Experiment_test'

    path_bundle = [no_acceptor_path, no_donor_path, no_cell_path, sample_path_list, data_root]

    test_experiment = Experiment('test_experiment_00', path_bundle)
    test_experiment.background_adjustments_calc()
    test_experiment.define_bt()

    test_img = SampleImage(
        experiment = test_experiment,
        sample_path= 'Experiment_test/Uninfected/Vinculin-TL',
        filename= test_experiment.samples_dict[
            'Experiment_test/Uninfected/Vinculin-TL'][0])

    unittest.main()
