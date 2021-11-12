import numpy as np

from utils import get_files, load_image


def set_background(file_path, reset=True, bands=None):
    """Loads a image files as background. Unless specified, also provides a fresh label mask and blocked_area mask.

        Keyword arguments:
        file_path -- path to the image file
        reset -- resets mask and blocked_area (default True)
        bands -- sets bands to specified value, leaves as is if None (default None)
    """
    background = load_image(file_path, bands)
    if reset:
        mask = np.zeros_like(background)
        blocked_area = np.zeros_like(background)
        return background, mask, blocked_area
    else:
        return background


def get_trees(files_path, file_type=None):
    """Creates a dictionary containing all image file paths in a given folder.
    File type can be specified, defaults to .tif

            Keyword arguments:
            files_path -- path to folder containing tree image files
            file_type -- type of tree image files, defaults to 'tif' (default None)
    """
    if file_type is None:
        file_type = 'tif'
    files = get_files(files_path, file_type)
    trees = {}
    for idx, file in enumerate(files):
        tree_type = str(file).rsplit('_', 1)[0]
        trees[idx] = (file, tree_type)
    print(f'Successfully loaded {len(trees)} tree images of type {file_type}.')
    return trees
