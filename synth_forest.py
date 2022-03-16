import warnings
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import rescale, resize, downscale_local_mean

from bezier_shape import random_shape
from utils import *

area_per_pixel = 0
tree_counter = 0
tree_type_counter = {}
verbose = False
background = np.empty(0)
mask = np.empty(0)
free_area = np.empty(0)
height_mask = np.empty(0)
edge_mask = np.empty(0)
type_to_number = {}
number_to_type = {}
trees = {'tree_type': [], 'file': []}

tree_type_grouping = {"BAH": "BAH",  # SHL
                      "BI": "BI", "GBI": "BI",  # BI
                      "BU": "BU", "RBU": "BU",  # BU
                      "EI": "EI", "SE": "EI", "SEI": "EI", "TEI": "EI",  # EI
                      "ELA": "ELA",  # LA
                      "ER": "ER", "RER": "ER", "SER": "ER",  # ER
                      "FI": "FI", "FIS": "FI", "GFI": "FI", "OFI": "FI", "PFI": "FI", "SFI": "FI",  # FI
                      "GES": "GES", "ES": "GES",  # SHL
                      "HBU": "HBU",
                      "KI": "KI", "GKI": "KI", "WKI": "KI",  # KI
                      "REI": "REI",  # EI
                      "SHL": "SHL", "AH": "SHL", "ROB": "SHL",  # SHL
                      "SKI": "SKI",
                      "SWL": "SWL", "ASP": "SWL", "PAP": "SWL",  # SWL
                      "WLI": "WLI",  # SWL
                      }

tree_type_likelihood = {"BAH": 1,
                        "BI": 5,
                        "BU": 4,
                        "EI": 1,
                        "ELA": 3,
                        "ER": 2,
                        "FI": 7,
                        "GES": 1,
                        "HBU": 1,
                        "KI": 6,
                        "REI": 1,
                        "SHL": 1,
                        "SKI": 1,
                        "SWL": 1,
                        "WLI": 1
                        }

tree_type_likelihood = {k: v / total for total in (sum(tree_type_likelihood.values()),)
                        for k, v in tree_type_likelihood.items()}


def set_background(file_path, pixel_area=1, augment=False, bands=None, reset=True):
    """Loads a image files as background. Unless specified, also provides a fresh label mask and blocked_area mask.

        Keyword arguments:
        file_path -- path to the image file
        pixel_area -- area of a single square pixel in m² (default 1)
        augment -- if the background image should be augmented (default False)
        bands -- sets bands to specified value, leaves as is if None (default None)
        reset -- resets mask and blocked_area (default True)
    """
    global background, mask, free_area, height_mask, edge_mask
    if not str(file_path).endswith('.tif'):
        file_path = np.random.choice(get_files(file_path, 'tif'))
    background = load_image(file_path, bands)
    if augment:
        background = background_augmentation(background)
    if reset:
        global tree_counter, area_per_pixel
        area_per_pixel = pixel_area
        tree_counter = 0
        mask = np.zeros_like(background[:, :, 0])
        free_area = np.ones_like(background[:, :, 0])
        height_mask = np.ones_like(background[:, :, 0], dtype='int32')
        edge_mask = np.zeros_like(background[:, :, 0])
        return mask, free_area, height_mask, edge_mask
    else:
        return 0


def get_trees(files_path, file_type=None):
    """Creates a dictionary containing all image file paths in a given folder.
    File type can be specified, defaults to .tif

            Keyword arguments:
            files_path -- path to folder containing tree image files
            file_type -- type of tree image files, defaults to 'tif' (default None)
    """
    global type_to_number, number_to_type, trees
    if file_type is None:
        file_type = 'tif'
    files = get_files(files_path, file_type)
    tree_types = ['background']
    for file in files:
        tree_type = str(file).rsplit('\\', 1)[-1].rsplit('/', 1)[-1].rsplit('_', 1)[0].rsplit('_', 1)[1].upper()
        if tree_type in tree_type_grouping.keys():
            tree_type = tree_type_grouping[tree_type]
        else:
            warnings.warn(f'{tree_type} not in known tree types: {list(tree_type_grouping.keys())}.'
                          f' A new class will be created.')
        trees['tree_type'].append(tree_type)
        trees['file'].append(file)
        if tree_type not in tree_types:
            tree_types.append(tree_type)

    trees = pd.DataFrame(trees)

    tree_labels = np.arange(len(tree_types), dtype='uint8')
    type_to_number = dict(zip(tree_types, tree_labels))
    number_to_type = dict(zip(tree_labels, tree_types))

    if verbose:
        print(f'Successfully loaded {len(trees)} tree images of type {file_type}.')
    return trees, type_to_number, number_to_type


def place_tree(distance, area=None, augment=True, cluster=False, tight=False, tree_type=None):
    """Places a single tree in a given distance of all other trees, updates the image mask and free area respectively.

            Keyword arguments:
            distance -- distance in pixels to be blocked around each tree
            area -- an area mask specifying where trees can be placed (default None --> uses internal free mask)
            augment -- if the tree image should be augmented (default True)
            cluster -- if trees are placed in a tree cluster or in a free space (default False)
            tight -- if trees should be placed in a tight layout (default False)
    """
    global background, mask, height_mask, edge_mask
    if distance != 0:
        rnd_distance = np.random.normal(distance, distance)
        distance = int(np.clip(rnd_distance, distance * 0.5, distance * 1.5) / np.sqrt(area_per_pixel))
    if area is None:
        area = free_area
    # if kernel_ratio is None:
    #     kernel_ratio = 1
    # else:
    #     kernel_ratio = int(np.round(1 / kernel_ratio, 0))

    if tree_type is None:
        tree, tree_type, height = random_tree(trees, augment)  # selects a tree at random from a list of trees
    else:
        p = 0.9  # probability, that same tree will be placed again
        if not np.random.uniform() > p:  # so a likelihood of p
            tree, tree_type, height = random_tree(trees.loc[trees['tree_type'] == tree_type], augment)
        else:  # 1 - p
            tree, tree_type, height = random_tree(trees.loc[trees['tree_type'] != tree_type], augment)
    tree_label = type_to_number[tree_type]  # converts tree_type to label

    kernel_ratio = 1

    place = False
    while not place:
        kernel = np.int64(fill_contours(tree[:, :, 0] != 0))
        kernel = np.int64(downscale_local_mean(kernel, (kernel_ratio, kernel_ratio)) != 0)
        area_with_buffer = np.int64(convolve2d(area == 0, kernel, mode='same') > 0) == 0

        if np.sum(area_with_buffer) == 0:
            # if verbose:
            #    print('\nImage does not contain any free area anymore. No tree was placed.')
            if cluster and kernel_ratio < 5:
                kernel_ratio += 1
                tight = False
            else:
                return 1
        else:
            x, y = random_position(area_with_buffer)
            place = True

    if tight:
        direction = np.random.choice(4)
        contact = False
        pos = [x, y]
        while not contact:
            pos[int(direction / 2)] += direction % 2 * -2 + 1
            if pos[0] > background.shape[0] or pos[1] > background.shape[1] or pos[0] < 0 or pos[1] < 0:
                x = np.clip(pos[0], 0, background.shape[0])
                y = np.clip(pos[1], 0, background.shape[1])
                break
            if area_with_buffer[x, y] == 0:
                contact = True
            else:
                x, y = pos

    boundaries = background.shape

    x_area, y_area, tree = set_area(x, y, tree, boundaries)  # sets image area, crops if necessary

    background, mask, height_mask, edge_mask = place_in_background(tree, tree_label, x_area, y_area, height,
                                                                   background, mask, height_mask, edge_mask)

    if distance == 0:
        shape_type = 'close'
        distance = int(np.mean(tree.shape[:2]) / 2)
    else:
        shape_type = 'single_tree'

    if cluster:
        block_mask = fill_contours(tree[:, :, 0] != 0)
    else:
        block_mask = random_shape(distance * 2, shape_type)

    x_block_area, y_block_area, block_mask = set_area(x, y, block_mask, boundaries)

    area[x_block_area[0]:x_block_area[1], y_block_area[0]:y_block_area[1]] *= block_mask == 0  # sets blocked area

    global tree_counter, tree_type_counter
    tree_counter += 1
    if tree_type not in tree_type_counter.keys():
        tree_type_counter[tree_type] = 0
    tree_type_counter[tree_type] += 1

    return 0


def fill_with_trees(distance, area=None, cluster=False, fixed_distance=True):
    """Repeats the 'place_tree'-function until no more trees can be placed.

                Keyword arguments (same as 'place_tree'):
                distance -- distance in pixels to be blocked around each tree (irrelevant if fixed_distance = True)
                area -- area to be given to place_tree (default None --> uses internal free mask)
                cluster -- if a cluster or a forest is being filled (default False)
                fixed_distance -- if distance should be fixed to distance value or depending on tree size (default True)
    """
    if not fixed_distance:
        distance = 0

    if cluster:
        tree_type = np.random.choice(list(tree_type_likelihood.keys()), p=list(tree_type_likelihood.values()))
    else:
        tree_type = None

    full = False
    counter = 0

    while not full:
        full = place_tree(distance, area, cluster=cluster, tight=cluster, tree_type=tree_type)
        if not full:
            counter += 1

    if cluster and verbose:
        print(f'\nCluster has been filled. A total of {counter} trees have been placed within the cluster.')
    elif verbose:
        print(f'\nForest has been filled. A total of {counter} additional trees have been placed.')


def place_cluster(area, area_in_pixel=False):
    """Creates a random shape of size 'area'. This shape is then filled with trees using the fill_trees function.

                Keyword arguments (same as 'place_tree'):
                area -- desired area of cluster in m² (unless area_in_pixel = True)
                area_in_pixel -- if area is provided in pixel or in m² (default False)
    """
    global background, free_area, edge_mask
    cluster_mask = random_shape(np.min([background.shape[0], background.shape[1]]), shape_type='cluster')
    cluster_area = np.sum(cluster_mask)

    if not area_in_pixel:
        area = int(area / area_per_pixel)

    if cluster_area < area:
        if verbose:
            print('Area to large to form a cluster in image. '
                  f'Area will be reduced to {int(cluster_area * area_per_pixel)}m².')
        area = cluster_area

    area_ratio = np.sqrt(area / cluster_area)

    cluster_mask = rescale(cluster_mask, area_ratio)
    block_mask = rescale(cluster_mask, 1.3)

    boundaries = background.shape

    x_range = np.arange(boundaries[0])
    y_range = np.arange(boundaries[1])
    x_range = np.clip(x_range, cluster_mask.shape[0] // 2, boundaries[0] - cluster_mask.shape[0] // 2)
    y_range = np.clip(y_range, cluster_mask.shape[1] // 2, boundaries[1] - cluster_mask.shape[1] // 2)

    x = np.random.choice(x_range)
    y = np.random.choice(y_range)

    x_area, y_area, cluster_mask = set_area(x, y, cluster_mask, boundaries)

    temporary_area = np.zeros_like(free_area)
    temporary_area[x_area[0]:x_area[1], y_area[0]:y_area[1]] = cluster_mask

    contours, hierarchy = cv2.findContours(np.uint8(temporary_area), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # gets contours of the tree
    try:
        x_coords = contours[0][:, :, 1].flatten() + x_area[0]  # x coordinates of the contours
        y_coords = contours[0][:, :, 0].flatten() + y_area[0]  # y coordinates of the contours

        edge_mask[x_coords, y_coords] = 1
    except IndexError:
        pass

    x_area, y_area, block_mask = set_area(x, y, block_mask, boundaries)

    background = np.multiply(background.astype('float64'),
                             np.expand_dims(temporary_area, axis=2) * 0.3
                             + np.int64(np.expand_dims(temporary_area, axis=2) == 0))
    background = np.round(background, 0).astype('uint8')

    fill_with_trees(0, temporary_area, cluster=True)

    free_area[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= block_mask == 0


def dense_forest():
    """Places a shadow on the full background and fills the image with trees."""
    global background
    background = np.multiply(background.astype('float64'), 0.3)
    background = np.round(background, 0).astype('uint8')
    fill_with_trees(0, cluster=True)


def forest_edge():
    """Creates a single cluster at least 4 times as big as the image.
    Moves the cluster in a random direction (up/down/left/right),
    then fills it with trees and adds sparse trees around the created forest border."""
    global background, free_area
    cluster_mask = random_shape(np.max([background.shape[0], background.shape[1]]) * 2, shape_type='close')

    boundaries = background.shape

    side = np.random.choice(4)
    if side in [0, 1]:  # down or up, x-direction
        move_distance = int(boundaries[0] / 2) \
                        - np.random.choice(np.arange(- 4 * boundaries[0] // 10, 4 * boundaries[0] // 10))
        if side == 1:  # up side, down movement
            x_range_cluster = [0, move_distance]
        else:  # down side, up movement
            x_range_cluster = [cluster_mask.shape[0] - move_distance, cluster_mask.shape[0]]
        y_range_cluster = [int((cluster_mask.shape[1] - boundaries[1]) / 2),
                           int((cluster_mask.shape[1] - boundaries[1]) / 2) + boundaries[1]]
    else:  # left or right, y-direction
        move_distance = int(boundaries[1] / 2) \
                        - np.random.choice(np.arange(- 4 * boundaries[1] // 10, 4 * boundaries[1] // 10))
        if side == 2:  # left side, right movement
            y_range_cluster = [0, move_distance]
        else:  # right side, left movement
            y_range_cluster = [cluster_mask.shape[1] - move_distance, cluster_mask.shape[1]]
        x_range_cluster = [int((cluster_mask.shape[0] - boundaries[0]) / 2),
                           int((cluster_mask.shape[0] - boundaries[0]) / 2) + boundaries[0]]

    cluster_mask = cluster_mask[x_range_cluster[0]:x_range_cluster[1], y_range_cluster[0]:y_range_cluster[1]]
    temporary_area = np.zeros_like(free_area)
    distance = int(5 / np.sqrt(area_per_pixel))

    if side in [0, 1]:  # down or up, x-direction
        block_mask = resize(cluster_mask, (cluster_mask.shape[0] + distance, cluster_mask.shape[1]))
        if side == 1:  # up side, down movement
            temporary_area[-cluster_mask.shape[0]:] = cluster_mask
            if block_mask.shape[0] > boundaries[0]:
                block_mask = block_mask[-boundaries[0]:]
            free_area[-block_mask.shape[0]:] *= block_mask == 0
        else:  # down side, up movement
            temporary_area[:cluster_mask.shape[0]] = cluster_mask
            if block_mask.shape[0] > boundaries[0]:
                block_mask = block_mask[:boundaries[0]]
            free_area[:block_mask.shape[0]] *= block_mask == 0
    else:  # left or right, y-direction
        block_mask = resize(cluster_mask, (cluster_mask.shape[0], cluster_mask.shape[1] + distance))
        if side == 2:  # left side, right movement
            temporary_area[:, boundaries[1] - cluster_mask.shape[1]:] = cluster_mask
            if block_mask.shape[1] > boundaries[1]:
                block_mask = block_mask[:, -boundaries[1]:]
            free_area[:, boundaries[1] - block_mask.shape[1]:] *= block_mask == 0
        else:  # right side, left movement
            temporary_area[:, :cluster_mask.shape[1]] = cluster_mask
            if block_mask.shape[1] > boundaries[1]:
                block_mask = block_mask[:, :boundaries[1]]
            free_area[:, :block_mask.shape[1]] *= block_mask == 0

    background[:, :, :3] = np.multiply(background.astype('float64')[:, :, :3],
                                       np.expand_dims(temporary_area, axis=2) * 0.3
                                       + np.int64(np.expand_dims(temporary_area, axis=2) == 0))
    background = np.round(background, 0).astype('uint8')

    fill_with_trees(0, temporary_area, cluster=True)


def tree_type_distribution(back=True):
    """Calculates distribution of class pixels in the image.

                Keyword arguments (same as 'place_tree'):
                back -- if the background pixels should be considered when calculating the distribution
    """
    tree_type_area = {}
    area = 0
    for number in tree_type_counter.keys():
        label = type_to_number[number]
        label_area = np.sum(mask == label)
        tree_type = number_to_type[label]
        tree_type_area[tree_type] = label_area
        area += label_area
    if back:
        label = type_to_number['background']
        label_area = np.sum(mask == label)
        tree_type = number_to_type[label]
        tree_type_area[tree_type] = label_area
        area += label_area
    for label in tree_type_area:
        tree_type_area[label] = np.round(tree_type_area[label] / area, 2)

    return tree_type_area


def finish_image():
    global background, edge_mask
    background = blur_edges(background, edge_mask)

    return background


def visualize():
    """Visualizes background, mask, height_mask, free_area if provided."""
    global background, mask, free_area, height_mask
    if background is not None:
        plt.imshow(background)
        plt.title('Synthetic image')
        plt.show()
    if mask is not None:
        plt.imshow(mask, cmap='hot')
        plt.title('Image mask')
        plt.show()
    if height_mask is not None:
        plt.imshow(height_mask, cmap='Greys_r')
        plt.title('Height map')
        plt.show()
    if free_area is not None:
        plt.imshow(free_area, cmap='Greys_r')
        plt.title('Free area')
        plt.show()


def detailed_results(show=False):
    """Prints numbers of trees as well as distribution."""
    if show:
        print(f'\nTotal number of trees placed: {tree_counter}.')
        print(f'Distribution of tree types: {tree_type_counter}.')
        print(f'Percentage: {tree_type_distribution()}.')
        print(f'Percentage without background: {tree_type_distribution(back=False)}.')

    return tree_counter, tree_type_counter, tree_type_distribution(), tree_type_distribution(back=False)
