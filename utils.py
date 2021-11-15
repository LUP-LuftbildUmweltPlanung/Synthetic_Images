import os
import glob
import cv2
import numpy as np
from pathlib import Path


def get_files(directory, file_type):
    """Returns a list of all files of the given type in the given directory."""
    directory = Path(directory)
    ori_dir = os.getcwd()
    os.chdir(directory)
    files = [directory / file for file in glob.glob('*.' + file_type)]
    os.chdir(ori_dir)
    return files


def load_image(path, bands=None):
    """Loads a single image file from a provided path. Reduces bands if bands variable is provided."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bands is None:
        return image
    else:
        return image[:, :, :bands]


def save_image(image, path):
    """Stores image in provided path."""
    cv2.imwrite(path, image)
    print('Image saved successfully.')
    
    
def select_random(trees):
    """Selects and returns a random tree from a list containing tuples."""
    tree_idx = np.random.choice(len(trees))
    tree = load_image(trees[tree_idx][0])
    tree_type = trees[tree_idx][1]
    return tree, tree_type


def set_area(x, y, tree, limits):
    tree_shape = tree.shape
    tree_center = (int(tree_shape[0] / 2), int(tree_shape[1] / 2))
    x_area = [x - tree_center[0],
               x - tree_center[0] + tree_shape[0]]
    y_area = [y - tree_center[1],
               y - tree_center[1] + tree_shape[1]]
    
    tree = crop_image(x_area, y_area, tree, limits)

    x_area = np.clip(x_area, 0, limits[0])
    y_area = np.clip(y_area, 0, limits[1])
    return x_area, y_area, tree
    
    
def crop_image(x_area, y_area, tree, limits):
    if x_area[0] < 0:  # checks, if image is out of left bound
        overlap = -1 * x_area[0]
        tree = tree[overlap:]
    elif x_area[1] > limits[0]:  # checks, if image is out of right bound
        overlap = x_area[1] - limits[0]
        tree = tree[:-overlap]

    if y_area[0] < 0:  # checks, if image is out of upper bound
        overlap = -1 * y_area[0]
        tree = tree[:, overlap:]
    elif y_area[1] > limits[1]:  # checks, if image is out of lower bound
        overlap = y_area[1] - limits[1]
        tree = tree[:, :-overlap]
    return tree
