"""

"""
# builtins
import os


# 3rd party Dependencies
from skimage import io, util
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage.filters import convolve
import scipy.misc


# local
from utils import *



class Experiment(object) :
    """
    Coordinates the events of the FRET focal adhesion analysis.

    Attributes

        mean_channel_background (arr): mean background intensity for channels
            of a 3-panel FRET experiment.
        fret_df (DataFrame): dataframe to record image and segment level
            information.
    """

    def __init__(self,
                 experiment_name,
                 paths_list,
                 b=10,
                 min_intensity_percentile = 0.05,
                 merger_threshold = 15,
                 min_segment_size = 5,
                 FA_segmentation_threshold = 750.0) :
        """
        Input
        |   * paths_list - the return of request_paths fn.

        Return
        |   * self
        """

        # paths to data
        self.experiment_name = experiment_name
        self.data_path = paths_list[4]
        self.no_acceptor_path = paths_list[0]
        self.no_donor_path = paths_list[1]
        self.no_cell_path = paths_list[2]
        self.samples_path_list = paths_list[3]

        # valid image filenames
        self.no_acceptor_filenames = keep_valid_image_names(self.no_acceptor_path)
        self.no_donor_filenames = keep_valid_image_names(self.no_donor_path)
        self.no_cell_filenames = keep_valid_image_names(self.no_cell_path)

        self.samples_dict = {}
        for sample_path in self.samples_path_list :
            self.samples_dict[sample_path] = keep_valid_image_names(sample_path)
        self.b = b
        self.th = min_intensity_percentile
        self.merger_th = merger_threshold
        self.min_segment_size = min_segment_size
        self.FA_seg_th = FA_segmentation_threshold
        self.exp_parameter_url = self.write_experiment_parameters_json()

        # run necessary functions
        self.background_adjustments_calc()
        self.define_bt()
        self.fret_df = self.initiate_fret_df()



    def background_adjustments_calc(self) :
        """
        Calculates the average background pixel intensity from n random fields
        per each image in the no_cell_path directory.

        Input
        |   * n - number of fields to assess for background pixel intensity

        Returns instance variable
        |   * self.mean_channel_background - 3-value list of the average pixel
        |       value in the background (no cell images).
        """
        channel_background_list = []
        for f in self.no_cell_filenames :
            fname = os.path.join(self.no_cell_path, f)
            img = io.imread(fname)

            channel_background_list.append(np.mean(np.mean(img, 0),0))

        self.mean_channel_background = np.mean(np.array(channel_background_list),0)



    def calculate_bleedthrough(self, control, bins, show_graphs = False) :
        """
        Function that calculates the FRET bleedthrough in control situations
        where no FRET should occur.

        Inputs
        |   * control - which control sample to calculate
        |   * bins - size of kernels for pooling step

        Returns
        |   * (slope, intercept) of the linear regression model fit to the
        |       processed control sample data.
        """

        # assertions
        assert control in ['no_acceptor_control', 'no_donor_control'], 'control parameter is not valid (`no_acceptor_control` or `no_donor_control` are valid options)'

        if control == 'no_acceptor_control' :
            channel = 0
            path_to_control = self.no_acceptor_path
            control_filenames = self.no_acceptor_filenames
        elif control == 'no_donor_control' :
            channel = 1
            path_to_control = self.no_donor_path
            control_filenames = self.no_donor_filenames

        # Calculate bleedthrough per image
        results = []
        for f in control_filenames :
            img = io.imread(os.path.join(path_to_control, f))
            assert (img.shape[0] % bins == 0) & (img.shape[1] % bins == 0), 'Image shape is not divisible by bin number'
            block_dims = [img.shape[0] // bins, img.shape[1] // bins] # // returns integer values

            control_blocks = util.view_as_blocks(img[:,:,channel], block_shape=tuple(block_dims))
            fret_blocks = util.view_as_blocks(img[:,:,2], block_shape = tuple(block_dims))

            for m in range(control_blocks.shape[1]) :
                for n in range(control_blocks.shape[0]) :
                    results.append( [np.mean(control_blocks[m,n]), np.mean(fret_blocks[m,n])])
        if show_graphs :
            plt.scatter([x[0] for x in results],
                        [y[1] for y in results])
            # TODO : label axes
            plt.show()


        LR_clf = LinearRegression()
        LR_clf.fit(X = np.array([x[0] for x in results]).reshape(-1,1),
                   y = np.array([y[1] for y in results]))

        return LR_clf.coef_[0], LR_clf.intercept_

    def define_bt(self, bins = 16, show_graphs = False) :
        """
        Wrapping function that combines `self.calculate_bleedthrough` calls to
        both control channels, storing returns in instance variables in the
        format of [coefficient, y_intercept].
        """
        self.dbt = self.calculate_bleedthrough(control = 'no_acceptor_control', bins = bins, show_graphs = show_graphs)
        self.abt = self.calculate_bleedthrough(control = 'no_donor_control', bins = bins, show_graphs = show_graphs)


    def initiate_fret_df(self) :
        """
        Utility function to initiate the fret_df to which image summary data
        will be added.
        """
        fret_df = pd.DataFrame([],
                                    columns = ['img_url',
                                                'exp_name',
                                                'exp_parameter_url',
                                                'FA_ID',
                                                'area',
                                                'mean_cFRET'],
                                    index = [])
        return fret_df

    def write_experiment_parameters_json(self):
        """
        Stores a json with the experiment parameters.
        """
        json_path = os.path.join(self.data_path, str(self.experiment_name)+'.json')
        par_dict = {'experiment': self.experiment_name,
                    'kernel_size': self.b,
                    'min_intensity_pc_threshold': self.th,
                    'merger_threshold': self.merger_th,
                    'min_segment_size': self.min_segment_size,
                    'FA_segmentation_threshold': self.FA_seg_th}

        with open(json_path, 'w') as fp :
            json.dump(par_dict, fp)

        return(os.path.join(self.data_path, str(self.experiment_name)+'.json'))



class SampleImage(object):
    """
    A placeholder object for single image calculations within `Experiment`.

    Attributes:
        experiment (object): an instance of `Experiment`
        sample_path (str): the full path and filename of instance image
        img (arr): an (m*n*3) array of the image (rgb)
        adj_img (arr): `SampleImage.img` with various adjustments
        master_dict (dictionary): dictionary containing the focal adhesion
            id keys with lists of pixel coordinates (flattened index).
        mask_arr (arr): mask of focal adhesion segments
        cFRET (arr): array containing the cFRET values for the image
    """

    def __init__(self, experiment, sample_path, filename) :
        """
        Initiates a SampleImage object.

        Inputs:
            experiment (object): an instance of `Experiment`
            sample_path (str): path to sample image
            filename (str): filename in the sample path dir
        """
        self.experiment = experiment
        self.fname = os.path.join(sample_path, filename)
        self.img = io.imread(self.fname)

        # make paths
        try:
            os.makedirs(os.path.join('Results', 'mask', sample_path))
        except :
            pass
        try:
            os.makedirs(os.path.join('Results', 'cFRET', sample_path))
        except:
            pass

        self.mask_url = os.path.join('Results', 'mask', self.fname)
        self.cFRET_url = os.path.join('Results', 'cFRET', self.fname)

        self.adj_img = self.threshold_filter()

        # segmentation
        self.waterfall_segmentation(arr = self.boxfilter(self.adj_img[:,:,1]))
        self.generate_mask(f_arr = self.adj_img[:,:,1])

        # cFRET calcs
        self.cFRET()
        self.img_fret_df = self.calculate_fret_stats()





    def threshold_filter(self) :
        """
        Sets pixel values that do not exceed a threshold arg to zero.

        Args:
            threshold (float; DEFAULT = 0.5): percentile of lowest pixel values
                to set at zero.
        Returns:
            adjusted_img (array): m*n*3 adjusted img array
        """
        adj_img = remove_maxouts(self.img)
        adj_img = self.subtract_background()

        max_pv = np.max(np.max(adj_img,0),0)
        th = max_pv.astype(float)*self.experiment.th

        for c in range(3):
            adj_img[:,:,c][adj_img[:,:,c] < th[c]] = 0
        self.adj_img = adj_img
        return self.adj_img

    def subtract_background(self) :
        """
        Subtracts the mean background pixel values calculated in
        `Experiment.background_adjustments_calc`

        Returns:
            adj_img (arr): image array with background intensity subtracted.
        """
        assert hasattr(self.experiment, 'mean_channel_background'), "`self.background_adjustments_calc` has not been run yet.  No background values exist for this experiment yet."
        adj_img = self.img.copy().astype(float)
        for c in range(3) :
            adj_img[:,:,c] -= self.experiment.mean_channel_background[c]
        return adj_img

    def boxfilter(self, img_arr) :
        """
        Function to convolve a local filter over an image.

        Args:
            arr (array): array for filter to be applied upon.  Typically the
                'acceptor' channel of a FRET 3-panel image.
            b (int): (DEFAULT = 10) the kernel size for convolution

        Returns:
            f_arr (array): filtered array.  Typically fed to the
                `SampleImage.waterfall_segmentation` function.
        """
        b = self.experiment.b
        inter_arr = convolve(img_arr, weights = np.ones([b,b]), mode='reflect', cval=0.0) / (b**2)
        f_arr = img_arr - inter_arr
        return f_arr

    def waterfall_segmentation(self, arr, verbose = True):
        """
        Function to id each FA segment.

        Args:
            arr (array): array for FA segmentation to be applied upon.
                Typically the array returned from `SampleImage.boxfilter`.
            I_threshold (float): minimum intensity to be considered for
                inclusion into a segment.
            merger_threshold (int): minimum number of pixels of a current
                segment required to avoid being merged into a neighoring
                segment when a collision occurs.
            min_pix_area (int): minimum size after algorithm to be considered an
                actual FA segment.
            verbose (bool): show print statements

        """
        merger_threshold = self.experiment.merger_th
        min_pix_area= self.experiment.min_segment_size
        I_threshold = self.experiment.FA_seg_th

        m, n = arr.shape
        assert m == n, 'Function not set up for non-square arrays'
        if verbose :
            print("Waterfall segmentation of FA initiated...")

        #reshape array
        flat_arr = arr.flatten(order='C')

        # assert indexing done correctly
        ROW_ix = 2
        COLUMN_ix = 3
        assert arr[ROW_ix,COLUMN_ix] == flat_arr[(ROW_ix*m) + COLUMN_ix]

        # rank pixel values in flat_arr
        rank_ix = np.argsort(flat_arr)[::-1].tolist()

        # shorten list to non-zero pixels
        ## find the index in the rank_ix where values in flat_arr become zero.
        for i, ix in enumerate(rank_ix) :
            if flat_arr[ix] <= I_threshold :
                cut_at_ix = i
                break
        ## cut the rank_ix
        rank_ix = rank_ix[:cut_at_ix]
        print('{} pixels passed intensity cutoff of {}'.format(
            len(rank_ix), I_threshold))

        id_dict = {}
        NEXT_SEGMENT_ID = 1

        while len(rank_ix) > 0 :

            ix = rank_ix.pop(0)
            pix_ix = get_neighbor_ix(ix, m)

            ix_has_id_list = []
            for neighbor in pix_ix :
                ix_has_id_list.append(check_neighbor_id(neighbor, id_dict))

            if np.sum(ix_has_id_list) == 0 :
                # no neighbors are labeled, assign a new FA id to this pixel.
                id_dict[NEXT_SEGMENT_ID] = [ix]
                NEXT_SEGMENT_ID += 1

            elif np.sum(ix_has_id_list) == 1 :
                # assign this pixel to the neighbor's id
                ## which neighbor is labeled?
                for neighbor in pix_ix :
                    if check_neighbor_id(neighbor, id_dict) :
                        ## retrive the FA segment id
                        segment_id = get_FA_seg_id(neighbor, id_dict)
                        ## assign the pixel under the FA segment id
                        id_dict[segment_id].append(ix)

            elif np.sum(ix_has_id_list) > 1 :
                ## which neighbors are labeled with an FA id?
                seg_FA_id_list = []
                for neighbor in pix_ix :
                    if check_neighbor_id(neighbor, id_dict) :
                        ## retrive the FA segment id
                        seg_FA_id_list.append(get_FA_seg_id(neighbor, id_dict))
                ## current pixel size of each FA id segment
                merge_list, keep_list = [], []
                for fa_id in set(seg_FA_id_list):
                    count = len(id_dict[fa_id])
                    ## determine if fa_id is big enough to be its own segment
                    if count < merger_threshold :
                        merge_list.append(fa_id)
                    else :
                        keep_list.append(fa_id)
                # NOTE : There are 3 results at this point:
                #   1.) length of merge_list is >0 and lenth of keep_list is <= 1.
                #   2.) length of merge_list is 0 and length of keep_list is >1.
                #   3.) length of merge_list is > 0 and the length of keep_list
                #       is >1.
                #   If option 1, then merge all fa_id pixel lists into a new
                #       fa_id, and assign the bridge pixel to this id.
                #   If option 2, assign the bridge pixel to the smaller(?) of
                #    the two segments.
                #   If option 3, (rare) merge the fa_id from the merge_list into
                #       the smaller(?)

                if (len(merge_list) > 0) and (len(keep_list)<=1) :
                    try :
                        new_seg_list = id_dict.pop(keep_list[0])
                    except :
                        new_seg_list = []
                    for fa_id in merge_list :
                        try :
                            new_seg_list.extend(id_dict.pop(fa_id))
                        except :
                            print(merge_list)
                            print(keep_list)
                            print(id_dict)
                            return "Error!"

                    id_dict[NEXT_SEGMENT_ID] = new_seg_list
                    ## add bridge pixel to the new fa_id
                    id_dict[NEXT_SEGMENT_ID].append(ix)
                    NEXT_SEGMENT_ID += 1

                elif (len(merge_list)==0) and (len(keep_list) >=1) :
                    tmp_count_dict = {}
                    for fa_id in keep_list :
                        tmp_count_dict.update({fa_id : len(id_dict[fa_id])})
                    smallest_id = min(tmp_count_dict, key=tmp_count_dict.get)
                    ## add bridge pixel to the smalled segment
                    id_dict[smallest_id].append(ix)

                elif (len(merge_list)>0) and (len(keep_list) >1) :
                    tmp_count_dict = {}
                    for fa_id in keep_list :
                        tmp_count_dict.update({fa_id : len(id_dict[fa_id])})
                    smallest_id = min(tmp_count_dict, key=tmp_count_dict.get)

                    new_seg_list = id_dict.pop(smallest_id)
                    for fa_id in merge_list :
                        new_seg_list.extend(id_dict.pop(fa_id))
                    id_dict[NEXT_SEGMENT_ID] = new_seg_list
                    ## add bridge pixel to the new fa_id
                    id_dict[NEXT_SEGMENT_ID].append(ix)
                    NEXT_SEGMENT_ID += 1
                else :
                    print("Error in merging process for pixel id {}".format(ix))
                    print(merge_list)
                    print(keep_list)
                    print(id_dict)
                    return "Error!"
        # filter for minimum pixel size
        master_dict = {}
        for x in id_dict :
            if len(id_dict[x]) > min_pix_area :
                master_dict[x] = id_dict[x]
        if verbose :
            print("\n{} focal adhesions identified by waterfall segmentation.".format(len(master_dict)))
        self.master_dict = master_dict

    def generate_mask(self, f_arr, show=False):
        """
        Generates a mask from a segmentation dictionary for visualization
        purposes.
        """

        mask_arr = np.zeros_like(f_arr)
        for fa_id in self.master_dict:
            for flat_ix in self.master_dict[fa_id] :
                mask_arr[get_coords(f_arr, flat_ix )] = fa_id
        self.mask_arr = mask_arr
        scipy.misc.toimage(mask_arr, cmin=0.0, cmax=3000).save(self.mask_url)
        if show :
            plt.imshow(self.mask_arr, cmap = 'nipy_spectral')
            plt.title(self.fname)
            plt.show()

    def cFRET(self):
        """
        Given an input image, calculates the cFRET value for each pixel.

        The cFRET is defined as the FRET channel value, less the FRET expected
        given the donor channel bleedthrough given the donor channel pixel
        intensity, less the FRET expected given the acceptor channel
        bleedthrough given the acceptor channel pixel intensity.

        cFRET = FRET - dbt(I_donor) - abt(I_acceptor),
        where FRET is the adjusted pixel intensity for the FRET channel,
        dbt and abt are the spectral bleedthrough functions and I_donor,
        I_acceptor are the adjusted intensity values for the donor and acceptor
        channels.

        Returns
            cFRET (arr): greyscale array of cFRET values
        """

        work_img = self.adj_img

        # split channels
        fret = work_img[:,:,2].astype(float)
        donor = work_img[:,:,0].astype(float)
        acceptor = work_img[:,:,1].astype(float)

        adj_donor = (donor*self.experiment.dbt[0]) - self.experiment.dbt[1]
        adj_acceptor = (acceptor*self.experiment.abt[0]) - self.experiment.abt[1]

        self.cFRET = np.clip(fret - adj_donor - adj_acceptor,  a_min = 0, a_max = None)
        scipy.misc.toimage(self.cFRET, cmin=0.0, cmax=3000).save(self.cFRET_url)


    def calculate_fret_stats(self) :
        """
        Calculates the focal adhesion metrics per segment id in an image and
        corresponding segment mask array.
        """

        tmp_df = self.experiment.initiate_fret_df()

        mask_arr = self.mask_arr


        # iterate through the segment ids
        for fa in self.master_dict :
            tmp_dict = {'img_url': self.fname,
                        'exp_name:': self.experiment.experiment_name,
                        'exp_parameter_url': self.experiment.exp_parameter_url,
                        'FA_ID': int(fa),
                        'area' : np.sum(mask_arr == fa),
                        'mean_cFRET': np.mean(self.cFRET[mask_arr == fa]) }
            tmp_df = tmp_df.append(pd.Series(tmp_dict), ignore_index=True)
        return tmp_df


if __name__ == '__main__' :
    pass
