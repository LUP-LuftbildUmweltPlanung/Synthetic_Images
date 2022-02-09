import glob
import os
from pathlib import Path

from scipy.ndimage.filters import gaussian_filter
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


def save_image(path, image, mask, verbose=False):
    """Stores image in provided path."""
    Path(str(path).rsplit('/', 1)[0].rsplit('\\', 1)[0]).mkdir(parents=True, exist_ok=True)
    path = Path(path)
    cv2.imwrite(str(path), image)
    mask_path = str(path).rsplit('.', 1)[0] + '_mask.' + str(path).rsplit('.', 1)[1]
    cv2.imwrite(mask_path, mask)
    if verbose:
        print('\nImage and mask saved successfully.')


def random_tree(trees, augment=False):
    """Selects and returns a random tree from a list containing tuples."""
    tree_data = trees.sample()
    tree = load_image(tree_data['file'].item())
    tree_type = tree_data['tree_type'].item()
    if augment:
        tree = tree_augmentation(tree)
    height = tree.shape[0] + tree.shape[1]  # Could probably be better
    return tree, tree_type, height


def tree_augmentation(tree):
    """Performs image augmentations on a provided image.
    Augmentations are: GridDistortion, Flip, Rotate, RandomScale."""
    transform = A.Compose([
        # A.RandomBrightnessContrast(p=0.3),
        # A.GridDistortion(p=0.3, distort_limit=(-0.1, 0.1),
        #                  interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
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


def place_in_background(tree, tree_label, x_area, y_area, height, background, mask, height_mask, edge_mask):
    """Places a single tree with the provided label at the provided position in both background and mask."""
    tree_mask = tree != 0  # mask to only remove tree part of image
    tree_mask[height_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] > height] = 0
    tree_mask = fill_contours(tree_mask)

    contours, hierarchy = cv2.findContours(np.uint8(tree_mask[:, :, 0]), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # gets contours of the tree

    try:
        x_coords = contours[0][:, :, 1].flatten() + x_area[0]  # x coordinates of the contours
        y_coords = contours[0][:, :, 0].flatten() + y_area[0]  # y coordinates of the contours

        edge_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0
        edge_mask[x_coords, y_coords] = 1
    except IndexError:
        pass

    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask == 0  # empties tree area in background
    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree * tree_mask  # adds tree into freshly deleted area

    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree_mask[:, :, 0] * np.array(tree_label).astype('uint8')
    # adds tree mask

    height_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    height_mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree_mask[:, :, 0] * height  # adds tree mask

    return background, mask, height_mask, edge_mask


def blur_edges(background, edge_mask):

    # OLD METHOD #
    # contour_array_tree = np.zeros_like(tree_mask[:, :, 0])
    # contour_array_tree[np.uint8(x_coords), np.uint8(y_coords)] = 1  # tree-sized array containing the tree contours
    #
    # contour_array_full = np.zeros_like(background[:, :, 0])
    # contour_array_full[x_area[0]:x_area[1], y_area[0]:y_area[1]] = contour_array_tree
    # # image-sized array containing the tree contours
    #
    # kernel = np.ones((3, 3))
    # contour_array_full = np.int64(convolve2d(contour_array_full, kernel, mode='same') > 0)  # buffered contour
    # contour_array_full = np.stack([contour_array_full]*background.shape[2], axis=2)  # adds image depth
    #
    # sigma = 2
    # window_size = 3
    # truncate = (((window_size - 1) / 2) - 0.5) / sigma
    # smooth_background = gaussian_filter(background, sigma=sigma, truncate=truncate, mode='nearest')
    #
    # background = np.where(contour_array_full, smooth_background, background)
    # OLD METHOD END #

    x_coords, y_coords = np.where(edge_mask == 1)

    window_size = 3
    sigma = 2
    truncate = (((window_size - 1) / 2) - 0.5) / sigma
    for x, y in zip(x_coords, y_coords):
        x = [np.max([0, x - int(window_size / 2)]), np.min([background.shape[0], x + int(window_size / 2)])]
        y = [np.max([0, y - int(window_size / 2)]), np.min([background.shape[1], y + int(window_size / 2)])]

        for layer in range(background.shape[2]):
            background[x[0]:x[1], y[0]:y[1], layer] = gaussian_filter(background[x[0]:x[1], y[0]:y[1], layer],
                                                                      sigma=sigma, truncate=truncate, mode='nearest')

    return background


def fill_contours(arr):
    """Fills a contour in a 1D array with ones."""
    return np.all([np.maximum.accumulate(arr, 1),
                   np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1],
                   np.maximum.accumulate(arr[::-1, :], 0)[::-1, :],
                   np.maximum.accumulate(arr, 0)], axis=0)


def unpack_results(result, image_count):
    """Unpacks the results returned from the multiprocessing Pool-function,
    running several instances of image creation at the same time."""
    labels_and_paths = []
    tot_trees = 0
    tot_tree_types = {}
    tot_tree_types_dist = {'background': 0}
    tot_tree_types_dist_no_back = {}
    for res in result:
        label, img_path, trees, tree_types, tree_types_dist, tree_types_dist_no_back = res
        labels_and_paths.append((label, img_path))
        tot_trees += trees
        for k in list(tree_types.keys()):
            if k not in tot_tree_types.keys():
                tot_tree_types[k] = tree_types[k]
            else:
                tot_tree_types[k] += tree_types[k]

        for k in list(tree_types_dist.keys()):
            if k not in tot_tree_types_dist.keys():
                tot_tree_types_dist[k] = tree_types_dist[k]
            else:
                tot_tree_types_dist[k] += tree_types_dist[k]

        for k in list(tree_types_dist_no_back.keys()):
            if k not in tot_tree_types_dist_no_back.keys():
                tot_tree_types_dist_no_back[k] = tree_types_dist_no_back[k]
            else:
                tot_tree_types_dist_no_back[k] += tree_types_dist_no_back[k]

        tot_tree_types_dist['background'] += tree_types_dist['background']

    for k in list(tot_tree_types.keys()):
        tot_tree_types_dist[k] = np.divide(tot_tree_types_dist[k], image_count).round(decimals=3)
        tot_tree_types_dist_no_back[k] = np.divide(tot_tree_types_dist_no_back[k], image_count).round(decimals=3)
    tot_tree_types_dist['background'] = np.divide(tot_tree_types_dist['background'], image_count).round(decimals=3)

    return labels_and_paths, tot_trees, tot_tree_types, tot_tree_types_dist, tot_tree_types_dist_no_back


def store_results(results, path):
    """Stores tree overview for an image group (e.g. Dense, Sparse etc.)."""
    with open(path / 'overview.txt', 'w') as file:
        for r in results:
            file.write(str(r))
            file.write('\n\n')


def merge_dictionaries(loaded_dict, new_dict):  # needs more testing
    """Merges two label dictionaries."""
    for k in new_dict.keys():
        if k not in loaded_dict.keys():
            loaded_dict[k] = len(loaded_dict.keys()) - 1
    return loaded_dict
