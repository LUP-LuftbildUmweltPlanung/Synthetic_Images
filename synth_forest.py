import warnings
import matplotlib.pyplot as plt

from utils import *
from bezier_shape import random_shape
from skimage.transform import rescale

area_per_pixel = 0
tree_counter = 0
tree_type_counter = {}
verbose = False


def set_background(file_path, pixel_area=1, augment=False, bands=None, reset=True):
    """Loads a image files as background. Unless specified, also provides a fresh label mask and blocked_area mask.

        Keyword arguments:
        file_path -- path to the image file
        pixel_area -- area of a single square pixel in m² (default 1)
        augment -- if the background image should be augmented (default False)
        bands -- sets bands to specified value, leaves as is if None (default None)
        reset -- resets mask and blocked_area (default True)
    """
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
        return background, mask, free_area, height_mask
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
    trees = []
    tree_types = ['background']
    for file in files:
        tree_type = str(file).rsplit('\\', 1)[1].rsplit('_', 1)[0]
        trees.append((file, tree_type))
        if tree_type not in tree_types:
            tree_types.append(tree_type)

    tree_labels = np.arange(len(tree_types), dtype='uint8')
    type_to_number = dict(zip(tree_types, tree_labels))
    number_to_type = dict(zip(tree_labels, tree_types))

    if verbose:
        print(f'Successfully loaded {len(trees)} tree images of type {file_type}.')
    return trees, type_to_number, number_to_type


def place_tree(distance, trees, background, mask, free_area, height_mask, type_to_number, augment=True):
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

    tree, tree_type, height = random_tree(trees, augment)  # selects a tree at random from a list of trees
    tree_label = type_to_number[tree_type]  # converts tree_type to label

    # BUFFER, IN PROGRESS #
    # from scipy.signal import convolve2d
    # kernel = np.int64(tree[:, :, 0] != 0)
    # free_area_with_buffer = np.int64(convolve2d(free_area == 0, kernel, mode='same') > 0)
    # BUFFER, IN PROGRESS #

    if np.sum(free_area) == 0:  # IF BUFFER REPLACE WITH FREE_AREA_WITH_BUFFER == 0
        # if verbose:
        #    print('\nImage does not contain any free area anymore. No tree was placed.')
        return 1

    x, y = random_position(free_area)  # selects a random position in the image

    boundaries = background.shape

    x_area, y_area, tree = set_area(x, y, tree, boundaries)  # sets image area, crops if necessary

    place_in_background(tree, tree_label, x_area, y_area, height, background, mask, height_mask)

    if distance == 0:
        shape_type = 'close'
        distance = int(np.mean(tree.shape[:2])/2)
    else:
        shape_type = 'single_tree'

    block_mask = random_shape(distance*2, shape_type)
    x_block_area, y_block_area, block_mask = set_area(x, y, block_mask, boundaries)

    free_area[x_block_area[0]:x_block_area[1], y_block_area[0]:y_block_area[1]] *= block_mask == 0  # sets blocked area

    global tree_counter, tree_type_counter
    tree_counter += 1
    if tree_type not in tree_type_counter.keys():
        tree_type_counter[tree_type] = 0
    tree_type_counter[tree_type] += 1

    return 0


def fill_with_trees(distance, trees, background, mask, free_area, height_mask, type_to_number, cluster=False,
                    fixed_distance=True):
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
    distance = int(distance / np.sqrt(area_per_pixel))

    if not fixed_distance:
        distance = 0

    full = False
    counter = 0
    while not full:
        full = \
            place_tree(distance, trees, background, mask, free_area, height_mask, type_to_number)
        if not full:
            counter += 1

    if cluster and verbose:
        print(f'\nCluster has been filled. A total of {counter} trees have been placed within the cluster.')
    elif verbose:
        print(f'\nForest has been filled. A total of {counter} additional trees have been placed.')


def place_cluster(area, trees, background, mask, free_area, height_mask,
                  type_to_number, area_in_pixel=False):
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

    block_mask = rescale(cluster_mask, area_ratio*1.5)
    cluster_mask = rescale(cluster_mask, area_ratio)

    boundaries = background.shape

    x_range = np.arange(boundaries[0])
    y_range = np.arange(boundaries[1])
    x_range = np.clip(x_range, cluster_mask.shape[0] // 2, boundaries[0] - cluster_mask.shape[0] // 2)
    y_range = np.clip(y_range, cluster_mask.shape[1] // 2, boundaries[1] - cluster_mask.shape[1] // 2)

    x = np.random.choice(x_range)
    y = np.random.choice(y_range)

    x_area, y_area, cluster_mask = set_area(x, y, cluster_mask, boundaries)

    temporary_free_area = np.zeros_like(mask)
    temporary_free_area[x_area[0]:x_area[1], y_area[0]:y_area[1]] = cluster_mask

    x_area, y_area, block_mask = set_area(x, y, block_mask, boundaries)

    free_area[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= block_mask == 0

    fill_with_trees(0, trees, background, mask, temporary_free_area,
                    height_mask, type_to_number, cluster=True)


def tree_type_distribution(mask, number_to_type, background=True):
    """Calculates distribution of class pixels in the image.

                Keyword arguments (same as 'place_tree'):
                mask -- array containing the image mask (1-dimensional)
                type_to_number -- dictionary mapping numerical label to tree type
                background -- if background class should be considered (default True)
    """
    tree_type_area = {}
    area = 0
    for label in range(mask.max() + 1):
        label_area = np.sum(mask == label)
        tree_type = number_to_type[label]
        if not background and tree_type == 'background':
            pass
        else:
            tree_type_area[tree_type] = label_area
            area += label_area
    for label in tree_type_area:
        tree_type_area[label] = np.round(tree_type_area[label] / area, 2)

    return tree_type_area


def visualize(background=None, mask=None, height_mask=None, free_area=None):
    """Visualizes background, mask, height_mask, free_area if provided."""
    if background is None and mask is None and height_mask is None and free_area is None:
        warnings.warn('Warning: No data provided. Visualization cancelled.')
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
        plt.title('Height map')
        plt.show()


def detailed_results(mask, number_to_type):
    """Prints numbers of trees as well as distribution."""
    print(f'\nTotal number of trees placed: {tree_counter}.')
    print(f'Distribution of tree types: {tree_type_counter}.')
    print(f'Percentage: {tree_type_distribution(mask, number_to_type)}.')
    print(f'Percentage without background: {tree_type_distribution(mask, number_to_type, background=False)}.')
