import warnings

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
type_to_number = {}
number_to_type = {}
trees = []

tree_type_grouping = {"GBI": "BI",  # BI
                      "BU": "BU", "HBU": "BU", "RBU": "BU",  # BU
                      "EI": "EI", "SE": "EI", "SEI": "EI", "TEI": "EI",  # EI
                      "REI": "REI",  # EI
                      "RER": "ER", "SER": "ER",  # ER
                      "FIS": "FI", "GFI": "FI", "OFI": "FI", "PFI": "FI", "SFI": "FI",  # FI
                      "GKI": "KI", "SKI": "KI", "WKI": "KI",  # KI
                      "ELA": "ELA",  # LA
                      "BAH": "BAH",  # SHL
                      "GES": "GES",  # SHL
                      "AH": "AH/ROB", "ROB": "AH/ROB",  # SHL
                      "ASP": "SWL", "PAP": "SWL", "SWL": "SWL",  # SWL
                      "WLI": "WLI"  # SWL
                      }


def set_background(file_path, pixel_area=1, augment=False, bands=None, reset=True):
    """Loads a image files as background. Unless specified, also provides a fresh label mask and blocked_area mask.

        Keyword arguments:
        file_path -- path to the image file
        pixel_area -- area of a single square pixel in m² (default 1)
        augment -- if the background image should be augmented (default False)
        bands -- sets bands to specified value, leaves as is if None (default None)
        reset -- resets mask and blocked_area (default True)
    """
    global background, mask, free_area, height_mask
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
        return mask, free_area, height_mask
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
        tree_type = str(file).rsplit('\\', 1)[-1].rsplit('/', 1)[-1].rsplit('_', 1)[0].upper()
        if tree_type in tree_type_grouping.keys():
            tree_type = tree_type_grouping[tree_type]
        else:
            warnings.warn(f'{tree_type} not in known tree types: {list(tree_type_grouping.keys())}.'
                          f' A new class will be created.')
        trees.append((file, tree_type))
        if tree_type not in tree_types:
            tree_types.append(tree_type)

    tree_labels = np.arange(len(tree_types), dtype='uint8')
    type_to_number = dict(zip(tree_types, tree_labels))
    number_to_type = dict(zip(tree_labels, tree_types))

    if verbose:
        print(f'Successfully loaded {len(trees)} tree images of type {file_type}.')
    return trees, type_to_number, number_to_type


def place_tree(distance, area=None, augment=True, cluster=False, tight=False, kernel_ratio=None):
    """Places a single tree in a given distance of all other trees, updates the image mask and free area respectively.

            Keyword arguments:
            distance -- distance in pixels to be blocked around each tree
            trees -- list containing tuples of type (tree_image_path, tree_type)
            background -- array containing the image (N-dimensional)
            mask -- array containing the image mask (1-dimensional)
            free_area -- array containing 1 where trees can be placed (1-dimensional)
            height_mask -- array containing the height mask (1-dimensional)
            type_to_number -- dictionary mapping tree type to numerical label
            augment -- if the tree image should be augmented (default True)
    """
    global background, mask, height_mask
    if distance != 0:
        rnd_distance = np.random.normal(distance, distance)
        distance = int(np.clip(rnd_distance, distance * 0.5, distance * 1.5) / np.sqrt(area_per_pixel))
    if area is None:
        area = free_area
    if kernel_ratio is None:
        kernel_ratio = 1
    else:
        kernel_ratio = int(np.round(1 / kernel_ratio, 0))

    tree, tree_type, height = random_tree(trees, augment)  # selects a tree at random from a list of trees
    tree_label = type_to_number[tree_type]  # converts tree_type to label

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

    background, mask, height_mask = place_in_background(tree, tree_label, x_area, y_area, height,
                                                        background, mask, height_mask)

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


# def multi_place_tree(data):
#     distance, area, cluster = data
#     return place_tree(distance, area, cluster=cluster, tight=cluster)


def fill_with_trees(distance, area=None, cluster=False, fixed_distance=True):
    """Repeats the 'place_tree'-function until no more trees can be placed.

                Keyword arguments (same as 'place_tree'):
                distance -- distance in pixels to be blocked around each tree
                trees -- list containing tuples of type (tree_image_path, tree_type)
                background -- array containing the image (N-dimensional)
                mask -- array containing the image mask (1-dimensional)
                free_area -- array containing 1 where trees can be placed (1-dimensional)
                height_mask -- array containing the height mask (1-dimensional)
                type_to_number -- dictionary mapping tree type to numerical label
                cluster -- if a cluster or a forest is being filled (default False)
                fixed_distance -- if distance should be fixed to distance value or depending on tree size (default True)
    """
    if not fixed_distance:
        distance = 0

    full = False
    counter = 0

    while not full:
        full = \
            place_tree(distance, area, cluster=cluster, tight=cluster)
        if not full:
            counter += 1

    if cluster and verbose:
        print(f'\nCluster has been filled. A total of {counter} trees have been placed within the cluster.')
    elif verbose:
        print(f'\nForest has been filled. A total of {counter} additional trees have been placed.')


def place_cluster(area, area_in_pixel=False):
    """Creates a random shape of size 'area'. This shape is then filled with trees using the fill_trees function.

                Keyword arguments (same as 'place_tree'):
                area -- desired area of cluster in m² (unless area_in_pixel=True)
                trees -- list containing tuples of type (tree_image_path, tree_type)
                background -- array containing the image (N-dimensional)
                mask -- array containing the image mask (1-dimensional)
                free_area -- array containing 1 where trees can be placed (1-dimensional)
                height_mask -- array containing the height mask (1-dimensional)
                type_to_number -- dictionary mapping tree type to numerical label
                area_in_pixel -- if area is provided in pixel or in m² (default False)
    """
    global background, free_area
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

    x_area, y_area, block_mask = set_area(x, y, block_mask, boundaries)

    background[:, :, :3] = np.multiply(background.astype('float64')[:, :, :3],
                                       np.expand_dims(temporary_area, axis=2) * 0.3
                                       + np.int64(np.expand_dims(temporary_area, axis=2) == 0))
    background = np.round(background, 0).astype('uint8')

    fill_with_trees(0, temporary_area, cluster=True)

    free_area[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= block_mask == 0


def dense_forest():
    global background
    background[:, :, :3] = np.multiply(background.astype('float64')[:, :, :3], 0.3)
    background = np.round(background, 0).astype('uint8')
    fill_with_trees(0, cluster=True)


def forest_edge():
    global background, free_area
    cluster_mask = random_shape(np.max([background.shape[0], background.shape[1]]) * 2, shape_type='close')

    boundaries = background.shape

    side = np.random.choice(4)
    if side in [0, 1]:  # down or up, x-direction
        move_distance = int(boundaries[0] / 2) - np.random.choice(np.arange(- 4 * boundaries[0]//10, boundaries[0]//2))
        if side == 1:  # up side, down movement
            x_range_cluster = [0, move_distance]
        else:  # down side, up movement
            x_range_cluster = [cluster_mask.shape[0] - move_distance, cluster_mask.shape[0]]
        y_range_cluster = [int(cluster_mask.shape[1] / 4), int(cluster_mask.shape[1] / 4) + boundaries[1]]
    else:  # left or right, y-direction
        move_distance = int(boundaries[1] / 2) - np.random.choice(np.arange(- 4 * boundaries[1]//10, boundaries[1]//2))
        if side == 2:  # left side, right movement
            y_range_cluster = [0, move_distance]
        else:  # right side, left movement
            y_range_cluster = [cluster_mask.shape[1] - move_distance, cluster_mask.shape[1]]
        x_range_cluster = [int(cluster_mask.shape[0] / 4), int(cluster_mask.shape[0] / 4) + boundaries[0]]

    cluster_mask = cluster_mask[x_range_cluster[0]:x_range_cluster[1], y_range_cluster[0]:y_range_cluster[1]]
    temporary_area = np.zeros_like(free_area)
    distance = int(5 / np.sqrt(area_per_pixel))

    if side in [0, 1]:  # down or up, x-direction
        block_mask = resize(cluster_mask, (cluster_mask.shape[0] + distance, cluster_mask.shape[1]))
        if side == 1:  # up side, down movement
            temporary_area[boundaries[0] - cluster_mask.shape[0]:] = cluster_mask
            if block_mask.shape[0] > boundaries[0]:
                block_mask = block_mask[-boundaries[0]:]
            free_area[boundaries[0] - block_mask.shape[0]:] *= block_mask == 0
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
                block_mask = block_mask[:boundaries[1]]
            free_area[:, :block_mask.shape[1]] *= block_mask == 0

    background[:, :, :3] = np.multiply(background.astype('float64')[:, :, :3],
                                       np.expand_dims(temporary_area, axis=2) * 0.3
                                       + np.int64(np.expand_dims(temporary_area, axis=2) == 0))
    background = np.round(background, 0).astype('uint8')

    fill_with_trees(0, temporary_area, cluster=True)


def tree_type_distribution(back=True):
    """Calculates distribution of class pixels in the image.

                Keyword arguments (same as 'place_tree'):
                mask -- array containing the image mask (1-dimensional)
                type_to_number -- dictionary mapping numerical label to tree type
                background -- if background class should be considered (default True)
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
