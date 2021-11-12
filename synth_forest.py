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
        mask = np.zeros_like(background[:, :, 0])
        free_area = np.ones_like(background[:, :, 0])
        return background, mask, free_area
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
    print(type_to_number)
    print(number_to_type)

    print(f'Successfully loaded {len(trees)} tree images of type {file_type}.')
    return trees, type_to_number, number_to_type


def place_tree(distance, trees, background, mask, free_area, type_to_number):
    """Places a single tree in a given distance of all other trees, updates the image mask and free area respectively.

            Keyword arguments:
            distance -- distance in pixels to be blocked around each tree
            trees -- list containing tuples of type (tree_image_path, tree_type)
            background -- array containing the image (N-dimensional)
            mask -- array containing the image mask (1-dimensional)
            free_area -- array containing 1 where trees can be placed (1-dimensional)
            type_to_number -- dictionary mapping tree type to numerical label
    """
    if np.sum(free_area) == 0:
        print('Image does not contain any free area anymore. No tree was placed.')
        return background, mask, free_area, 1

    area_likelihoods = free_area / np.sum(free_area != 0)  # calculates likelihood for each position
    pos = np.random.choice(np.arange(free_area.size), p=area_likelihoods.flatten())  # selects position in 1D image

    x = int(pos / free_area.shape[1])  # calculates true x_position for 2D position from 1D position
    y = int(pos % free_area.shape[1])  # calculates true y_position for 2D position from 1D position

    tree_idx = np.random.choice(len(trees))  # selects random tree
    tree = load_image(trees[tree_idx][0])  # gets image
    tree_type = trees[tree_idx][1]  # gets tree type
    tree_label = type_to_number[tree_type]  # converts tree_type to label

    tree_shape = tree.shape  # gets image shape
    tree_center = (int(tree_shape[0] / 2), int(tree_shape[1] / 2))  # gets image center

    x_range = [x - tree_center[0],
               x - tree_center[0] + tree_shape[0]]  # gets x_range according do image shape and x position
    y_range = [y - tree_center[1],
               y - tree_center[1] + tree_shape[1]]  # gets y_range according do image shape and y position

    limits = background.shape

    if x_range[0] < 0:  # checks, if image is out of left bound
        overlap = -1 * x_range[0]
        tree = tree[overlap:]
    elif x_range[1] > limits[0]:  # checks, if image is out of right bound
        overlap = x_range[1] - limits[0]
        tree = tree[:-overlap]

    if y_range[0] < 0:  # checks, if image is out of upper bound
        overlap = -1 * y_range[0]
        tree = tree[:, overlap:]
    elif y_range[1] > limits[1]:  # checks, if image is out of lower bound
        overlap = y_range[1] - limits[1]
        tree = tree[:, :-overlap]

    tree_mask = tree != 0  # mask to only remove tree part of image

    x_range = np.clip(x_range, 0, limits[0])  # clip x_range to image bounds
    y_range = np.clip(y_range, 0, limits[1])  # clip y_range to image bounds

    background[x_range[0]:x_range[1], y_range[0]:y_range[1]] *= tree_mask == 0  # empties tree area in background
    background[x_range[0]:x_range[1], y_range[0]:y_range[1]] += tree  # adds tree into freshly deleted area

    mask[x_range[0]:x_range[1], y_range[0]:y_range[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    mask[x_range[0]:x_range[1], y_range[0]:y_range[1]] += tree_mask[:, :, 0] * tree_label  # adds tree mask

    x_block_range = np.clip([x - distance, x + distance], 0, limits[0])  # calculates blocked area from distance
    y_block_range = np.clip([y - distance, y + distance], 0, limits[1])

    free_area[x_block_range[0]:x_block_range[1], y_block_range[0]:y_block_range[1]] = 0  # sets blocked area

    return background, mask, free_area, 0


def fill_with_trees(distance, trees, background, mask, free_area, type_to_number):
    fill = 0
    counter = 0
    while fill == 0:
        background, mask, free_area, fill = place_tree(distance, trees, background, mask, free_area, type_to_number)
        counter += 1
        if counter % 50 == 0:
            print(f'{counter} trees placed.')

    print(f'Forest has been filled. A total of {counter} additional trees have been placed.')
