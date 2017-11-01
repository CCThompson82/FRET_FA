"""
Utility functions for FRET_FA module.
"""
# builtins
import os

# deps
import numpy as np
import matplotlib.pyplot as plt


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
    return no_acceptor_path, no_donor_path, no_cell_path, sample_path_list, data_root

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

def get_coords(arr, flat_ix) :
    """
    Utility to retrive pixel cordinates from a flattened index id.
    """
    m, n = arr.shape
    assert m == n , "Not set up for rectangle images"
    i = flat_ix // m
    j = flat_ix % m
    return i, j

def remove_maxouts(img) :
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

def show_image(cFRET, title=''):
    """
    show image
    """
    plt.imshow(cFRET, cmap = 'gray')
    plt.title(title)
    plt.show()
