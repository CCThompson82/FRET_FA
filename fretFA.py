"""

"""
# builtins
import os


# 3rd party Dependencies
from skimage import io, util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.ndimage.filters import convolve

def request_paths() :
    """
    Queries user to input the path directories containing relavent images
    for FRET analysis
    """
    print("="*80)
    print("Current working directory: {}".format(os.getcwd()))
    print("-"*80)
    data_root = input('path to experiment data?  ')
    print("-"*80)
    print('Setting up the control images')
    print("-"*80)
    for ix, folder in enumerate(os.listdir(data_root)) :
        print("[{}]  {}".format(ix, folder))

    no_acceptor_path = os.path.join(
        data_root,
        os.listdir(data_root)[int(
            input(
            "Please select the index number of the directory containing the 'no acceptor control' images:   "
                 ))])

    no_donor_path = os.path.join(
        data_root,
        os.listdir(data_root)[int(
            input(
            "Please select the index number of the directory containing the 'no donor control' images:   "
                 ))])

    no_cell_path = os.path.join(
        data_root,
        os.listdir(data_root)[int(
            input(
            "Please select the index number of the directory containing the 'no cell control' images :  "
                 ))])

    print('-'*80)
    print("{} 'no_acceptor_control' (.tif) images".format(np.sum( [x.find('.tif') != -1 for x in os.listdir(no_acceptor_path)])))
    print("{} 'no_donor_control' (.tif) images".format(np.sum( [x.find('.tif') != -1 for x in os.listdir(no_donor_path)])))
    print("{} 'no_cell_control' (.tif) images".format(np.sum( [x.find('.tif') != -1 for x in os.listdir(no_cell_path)])))
    print('-'*80)
    print('Setting up the sample images')
    print('-'*80)
    for ix, folder in enumerate(os.listdir(data_root)) :
        print("[{}]  {}".format(ix, folder))
    print("[F]  <Finished selecting samples>")
    print('-'*80)

    sample_path_list = []
    selection_loop = 'on'

    while selection_loop == 'on' :
        selection = input('Select an index containing (nested) sample images:  ').strip()
        if selection == 'F' :
            selection_loop = 'off'
            print('='*80)
        else :
            sample_dir = os.path.join(data_root, os.listdir(data_root)[int(selection)])

            for root, dirs, files in os.walk(sample_dir) :
                if (len(dirs) == 0) & (len(files) > 0) :
                    sample_path_list.append(root)
            print('-'*80)
    print( no_acceptor_path, no_donor_path, no_cell_path, sample_path_list)
    return no_acceptor_path, no_donor_path, no_cell_path, sample_path_list


def keep_valid_image_names(path) :
    """
    Cleaner function to drop non-image files from filename lists.
    """
    f_list = os.listdir(path)
    for idx, f in enumerate(f_list) :
        if f.find('.tif') == -1 :
            _ = f_list.pop(idx)
    return f_list


def get_neighbor_ix(ix, m) :
    """
    Utility to retrieve the index positions of surrounding pixels of
    ix from a 1D list
    """

    i = ix // m
    j = ix % m

    ret_list = []
    for r in range(-1,2,1) :
        for c in range(-1,2,1) :
            if ((r == 0) and (c == 0)) :
                pass
            else :
                ret_list.append( ((r+i)*m) + (j+c) )
    return ret_list

def check_neighbor_id(n_ix, id_dict) :
    """
    Utility to check if neibhbor has been assigned an id.
    """
    for fa_id in id_dict :
        if n_ix in id_dict[fa_id] :
            return True
    return False

def get_FA_seg_id(n_ix, id_dict) :
    """
    Utility to search FA_ids until the neighbor ix is found.
    Returns the FA id.
    """
    for fa_id in id_dict :
        if n_ix in id_dict[fa_id] :
            return fa_id







class Experiment(object) :

    def __init__(self, paths_list) :
        """
        Input
        |   * paths_list - the return of request_paths fn.

        Return
        |   * self
        """

        # paths to data
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
        return None

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
        return None

    def remove_maxouts(self, img) :
        """
        Zeros out any pixels at the maximum pixel value (16-bit = 2**16 - 1  )

        Inputs
        |   * img

        Returns
        |   * adjusted img array
        """
        adj_img = img.copy().astype(float)
        MAX = (2**16) - 1
        adj_img[ img == MAX] = 0
        return adj_img

    def subtract_background(self, img) :
        """
        Subtracts the mean background pixel values calculated in
        `self.background_adjustments_calc`

        Inputs
        |   * img

        Returns
        |   * adjusted img array
        """
        assert hasattr(self, 'mean_channel_background'), '`self.background_adjustments_calc` has not been run yet.  No background values exist for this experiment yet.'
        adj_img = img.copy().astype(float)
        for c in range(3) :
            adj_img[:,:,c] -= self.mean_channel_background[c]
        return adj_img

    def threshold_filter(self, img, threshold = 0.05) :
        """
        Sets pixel values that do not exceed a threshold arg to zero.

        Inputs
        |   * img
        |   * threshold - (DEFAULT = 0.05) float between 0,1

        Returns
        |   * adjusted img array
        """
        MAX_bit = (2**16)-1
        assert not np.any(img == MAX_bit), 'Maxout pixels not removed.  Run `self.remove_maxouts`.'

        adj_img = img.copy().astype(float)
        max_pv = np.max(np.max(adj_img,0),0)
        th = max_pv.astype(float)*threshold

        for c in range(3):
            adj_img[:,:,c][adj_img[:,:,c] < th[c]] = 0

        return adj_img


    def threshold_filter_gs(self, arr, threshold = 0.05) :
        """
        Sets pixel values that do not exceed a threshold arg to zero in an
        m x n array.

        Inputs
        |   * arr
        |   * threshold - (DEFAULT = 0.05) float between 0,1

        Returns
        |   * adjusted img array
        """
        MAX_bit = (2**16)-1
        #assert not np.any(img == MAX_bit), 'Maxout pixels not removed.  Run `self.remove_maxouts`.'

        adj_arr = arr.copy().astype(float)
        max_pv = np.max(adj_arr,0)
        th = max_pv.astype(float)*threshold

        adj_arr[adj_arr < th] = 0

        return adj_arr

    def cFRET(self, img):
        """
        Given an input image, calculates the cFRET value for each pixel.  The
        cFRET is defined as the FRET channel value, less the FRET expected given
        the donor channel bleedthrough given the donor channel pixel intensity,
        less the FRET expected given the acceptor channel bleedthrough given the
        acceptor channel pixel intensity.

        cFRET = FRET - dbt(I_donor) - abt(I_acceptor),
        where FRET is the adjusted pixel intensity for the FRET channel,
        dbt and abt are the spectral bleedthrough functions and I_donor,
        I_acceptor are the adjusted intensity values for the donor and acceptor
        channels.

        Inputs
        |   * img

        Returns
        |   * corrected FRET array (m,n)
        """
        assert hasattr(self, 'mean_channel_background'), "Run `self.background_adjustments_calc` fn."
        assert hasattr(self, 'dbt'), "Run `self.define_bt` function."
        assert hasattr(self, 'abt'), "Run `self.define_bt` function."

        # background adjustments
        ph_img = self.remove_maxouts(img)
        ph_img = self.subtract_background(ph_img)
        ph_img = self.threshold_filter(ph_img)

        # split channels
        fret = ph_img[:,:,2].astype(float)
        donor = ph_img[:,:,0].astype(float)
        acceptor = ph_img[:,:,1].astype(float)

        adj_donor = (donor*self.dbt[0]) - self.dbt[1]
        adj_acceptor = (acceptor*self.abt[0]) - self.abt[1]

        cFRET = np.clip(fret - adj_donor - adj_acceptor, a_min = 0, a_max = None)

        return cFRET



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
        return None

    def boxfilter(self, img_arr, b = 12) :
        """
        Function to convolve a local filter over an image.
        """

        inter_arr = convolve(img_arr, weights = np.ones([b,b]), mode='reflect', cval=0.0) / (b**2)
        return img_arr - inter_arr

    def waterfall_segmentation(self, arr,
                                     I_threshold = 0.5,
                                     merger_threshold = 9,
                                     min_pix_area=5,
                                     verbose = True):
        """
        Function to id each FA segment.
        """
        m, n = arr.shape
        assert m == n, 'Function not set up for non-square arrays'
        if verbose :
            print("Waterfall segmentation of FA initiated...")

        arr = self.threshold_filter_gs(arr, threshold = I_threshold)

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
            if flat_arr[ix] == 0.0 :
                cut_at_ix = i
                break
        ## cut the rank_ix
        rank_ix = rank_ix[:cut_at_ix]

        id_dict = {}
        NEXT_SEGMENT_ID = 1

        while len(rank_ix) > 0 :
            if verbose :
                if (len(rank_ix) % 500) == 0:
                    print('{} pixels left to assign'.format(len(rank_ix)))

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
                            return None

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
                    return None
        # filter for minimum pixel size
        master_dict = {}
        for x in id_dict :
            if len(id_dict[x]) > min_pix_area :
                master_dict[x] = id_dict[x]
        if verbose :
            print("\n{} focal adhesions identified by waterfall segmentation.".format(len(master_dict)))

        return master_dict



if __name__ == '__main__' :
    path_list = request_paths()
