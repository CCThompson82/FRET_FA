"""

"""
# builtins
import os


# 3rd party Dependencies
from skimage import io, util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

def check_data() :
    """Check validity of all data images and directories"""
    pass

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

        self.samples_filename_list = []
        for sample_path in self.samples_path_list :
            self.samples_filename_list.append(keep_valid_image_names(sample_path))
        return None




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


if __name__ == '__main__' :
    path_list = request_paths()
