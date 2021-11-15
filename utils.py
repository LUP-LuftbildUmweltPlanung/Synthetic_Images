import glob
import os
from pathlib import Path

import cv2
import numpy as np


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


def random_tree(trees):
    """Selects and returns a random tree from a list containing tuples."""
    tree_idx = np.random.choice(len(trees))
    tree = load_image(trees[tree_idx][0])
    tree_type = trees[tree_idx][1]
    return tree, tree_type


def set_area(x, y, tree, boundaries):
    """Calculates the boundaries of the insertion area."""
    tree_shape = tree.shape
    tree_center = (int(tree_shape[0] / 2), int(tree_shape[1] / 2))
    x_area = [x - tree_center[0],
              x - tree_center[0] + tree_shape[0]]
    y_area = [y - tree_center[1],
              y - tree_center[1] + tree_shape[1]]

    tree = crop_image(x_area, y_area, tree, boundaries)

    x_area = np.clip(x_area, 0, boundaries[0])
    y_area = np.clip(y_area, 0, boundaries[1])
    return x_area, y_area, tree


def crop_image(x_area, y_area, tree, boundaries):
    """Crops inserted image to boundaries of background image."""
    if x_area[0] < 0:  # checks, if image is out of left bound
        overlap = -1 * x_area[0]
        tree = tree[overlap:]
    elif x_area[1] > boundaries[0]:  # checks, if image is out of right bound
        overlap = x_area[1] - boundaries[0]
        tree = tree[:-overlap]

    if y_area[0] < 0:  # checks, if image is out of upper bound
        overlap = -1 * y_area[0]
        tree = tree[:, overlap:]
    elif y_area[1] > boundaries[1]:  # checks, if image is out of lower bound
        overlap = y_area[1] - boundaries[1]
        tree = tree[:, :-overlap]
    return tree


def random_position(free_area):
    """Randomly chooses a single point in the matrix using the free_area values as probabilities."""
    area_likelihoods = free_area / np.sum(free_area)  # calculates likelihood for each position
    pos = np.random.choice(np.arange(free_area.size), p=area_likelihoods.flatten())  # selects position in 1D image

    x = int(pos / free_area.shape[1])  # calculates true x_position for 2D position from 1D position
    y = int(pos % free_area.shape[1])  # calculates true y_position for 2D position from 1D position
    return x, y


def place_in_background(tree, tree_label, x_area, y_area, background, mask):
    """Places a single tree with the provided label at the provided position in both background and mask."""
    tree_mask = tree != 0  # mask to only remove tree part of image

    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask == 0  # empties tree area in background
    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree  # adds tree into freshly deleted area

    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree_mask[:, :, 0] * tree_label  # adds tree mask

    return background, mask


# Not really Utils, only moved here so synth_forest.py looks cleaner
def contact_position(area, cluster_mask, tree):
    """Randomly selects a contact point on the cluster and the tree and
    calculates the center position required for the tree for these two points to match."""
    side = np.random.choice(np.where(area > 0)[0])  # 0-left, 1-right, 2-up, 3-down
    contact = False
    if side == 0 or side == 1:
        x_pos = -1 * side  # so 0 or -1
        y_pos = np.random.choice(cluster_mask.shape[1])
        half = y_pos > cluster_mask.shape[1] / 2  # False: left half, True: right half

        while not contact:
            if cluster_mask[x_pos, y_pos] != 0:
                contact = True
            else:
                if x_pos >= 0:
                    x_pos += 1
                else:
                    x_pos -= 1
    else:
        x_pos = np.random.choice(cluster_mask.shape[0])
        y_pos = -1 * (side - 2)
        half = x_pos > cluster_mask.shape[0] / 2  # False: upper half, True: lower half

        while not contact:
            if cluster_mask[x_pos, y_pos] != 0:
                contact = True
            else:
                if y_pos >= 0:
                    y_pos += 1
                else:
                    y_pos -= 1

    if x_pos < 0:
        x_pos = cluster_mask.shape[0] + x_pos

    if y_pos < 0:
        y_pos = cluster_mask.shape[1] + y_pos

    cluster_contact = [x_pos, y_pos]

    if side == 0 or 2:
        tree_side = side + 1
    else:
        tree_side = side - 1

    contact = False
    if tree_side == 0 or tree_side == 1:
        x_pos = -1 * tree_side
        y_pos = np.random.choice(tree.shape[1] // 2)
        y_pos = y_pos + y_pos * half

        while not contact:
            if tree[x_pos, y_pos, 0] != 0:
                contact = True
            else:
                if x_pos >= 0:
                    x_pos += 1
                else:
                    x_pos -= 1
    else:
        x_pos = np.random.choice(tree.shape[0] // 2)
        y_pos = -1 * (tree_side - 2)
        x_pos = x_pos + x_pos * half

        while not contact:
            if tree[x_pos, y_pos, 0] != 0:
                contact = True
            else:
                if y_pos >= 0:
                    y_pos += 1
                else:
                    y_pos -= 1

    if x_pos < 0:
        x_pos = tree.shape[0] + x_pos

    if y_pos < 0:
        y_pos = tree.shape[1] + y_pos

    tree_contact = [x_pos, y_pos]

    x = area[0] + cluster_contact[0] + tree.shape[0] // 2 - tree_contact[0]
    y = area[2] + cluster_contact[1] + tree.shape[1] // 2 - tree_contact[1]

    return x, y
