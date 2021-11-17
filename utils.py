import glob
import os
from pathlib import Path

import albumentations as A
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


def save_image(path, image, mask):
    """Stores image in provided path."""
    path = Path(path)
    cv2.imwrite(str(path), image)
    mask_path = str(path).rsplit('.', 1)[0] + '_mask.' + str(path).rsplit('.', 1)[1]
    cv2.imwrite(mask_path, mask)
    print('\nImage and mask saved successfully.')


def random_tree(trees, augment=False):
    """Selects and returns a random tree from a list containing tuples."""
    tree_idx = np.random.choice(len(trees))
    tree = load_image(trees[tree_idx][0])
    tree_type = trees[tree_idx][1]
    if augment:
        tree = tree_augmentation(tree)
    height = tree.shape[0] + tree.shape[1]  # Could probably be better
    return tree, tree_type, height


def tree_augmentation(tree):
    """Performs image augmentations on a provided image.
    Augmentations are: GridDistortion, Flip, Rotate, RandomScale."""
    transform = A.Compose([
        # A.RandomBrightnessContrast(p=0.3),
        A.GridDistortion(p=0.3, distort_limit=(-0.1, 0.1),
                         interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
        A.Flip(p=0.5),
        A.Rotate(p=1.0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, limit=(-180, 180)),
        A.RandomScale(p=0.6, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.3, 0.3))
    ])
    return transform(image=tree)['image']


def background_augmentation(background):
    """Performs image augmentations on a provided image.
        Augmentations are: Flip."""
    transform = A.Compose([
        A.Flip(p=0.5),
        # A.Equalize(p=0.5),
        # A.GaussNoise(p=0.5),
        # A.RandomBrightnessContrast(p=0.5)
    ])
    return transform(image=background)['image']


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
    if x_area[1] > boundaries[0]:  # checks, if image is out of right bound
        overlap = x_area[1] - boundaries[0]
        tree = tree[:-overlap]

    if y_area[0] < 0:  # checks, if image is out of upper bound
        overlap = -1 * y_area[0]
        tree = tree[:, overlap:]
    if y_area[1] > boundaries[1]:  # checks, if image is out of lower bound
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


def place_in_background(tree, tree_label, x_area, y_area, height, background, mask, height_mask):
    """Places a single tree with the provided label at the provided position in both background and mask."""
    tree_mask = tree != 0  # mask to only remove tree part of image
    tree_mask[height_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] > height] = 0
    tree_mask = fill_contours(tree_mask)

    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask == 0  # empties tree area in background
    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree * tree_mask  # adds tree into freshly deleted area

    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree_mask[:, :, 0] * tree_label  # adds tree mask

    height_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    height_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree_mask[:, :, 0] * height  # adds tree mask


def fill_contours(arr):
    """Fills a contour in a 1D array with ones."""
    return np.all([np.maximum.accumulate(arr, 1),
                   np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1],
                   np.maximum.accumulate(arr[::-1, :], 0)[::-1, :],
                   np.maximum.accumulate(arr, 0)], axis=0)
