import numpy as np
from utils import get_files, load_image, select_random, set_area

tree_counter = 0
tree_type_counter = {}


def set_background(file_path, reset=True, bands=None):
    """Loads a image files as background. Unless specified, also provides a fresh label mask and blocked_area mask.

        Keyword arguments:
        file_path -- path to the image file
        reset -- resets mask and blocked_area (default True)
        bands -- sets bands to specified value, leaves as is if None (default None)
    """
    background = load_image(file_path, bands)
    if reset:
        global tree_counter
        tree_counter = 0
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
        print('\nImage does not contain any free area anymore. No tree was placed.')
        return background, mask, free_area, 1

    area_likelihoods = free_area / np.sum(free_area != 0)  # calculates likelihood for each position
    pos = np.random.choice(np.arange(free_area.size), p=area_likelihoods.flatten())  # selects position in 1D image

    x = int(pos / free_area.shape[1])  # calculates true x_position for 2D position from 1D position
    y = int(pos % free_area.shape[1])  # calculates true y_position for 2D position from 1D position

    tree, tree_type = select_random(trees)  # selects a tree at random from a list of trees
    tree_label = type_to_number[tree_type]  # converts tree_type to label

    limits = background.shape

    x_area, y_area, tree = set_area(x, y, tree, limits)  # sets image area, crops if necessary

    tree_mask = tree != 0  # mask to only remove tree part of image

    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask == 0  # empties tree area in background
    background[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree  # adds tree into freshly deleted area

    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] *= tree_mask[:, :, 0] == 0  # empties tree area in mask
    mask[x_area[0]:x_area[1], y_area[0]:y_area[1]] += tree_mask[:, :, 0] * tree_label  # adds tree mask

    x_block_area = np.clip([x - distance, x + distance], 0, limits[0])  # calculates blocked area from distance
    y_block_area = np.clip([y - distance, y + distance], 0, limits[1])

    free_area[x_block_area[0]:x_block_area[1], y_block_area[0]:y_block_area[1]] = 0  # sets blocked area

    global tree_counter
    tree_counter += 1

    global tree_type_counter
    if tree_type not in tree_type_counter.keys():
        tree_type_counter[tree_type] = 0
    tree_type_counter[tree_type] += 1

    return background, mask, free_area, 0


def fill_with_trees(distance, trees, background, mask, free_area, type_to_number, verbose=False):
    """Repeats the 'place_tree'-function until no more trees can be placed.

                Keyword arguments (same as 'place_tree'):
                distance -- distance in pixels to be blocked around each tree
                trees -- list containing tuples of type (tree_image_path, tree_type)
                background -- array containing the image (N-dimensional)
                mask -- array containing the image mask (1-dimensional)
                free_area -- array containing 1 where trees can be placed (1-dimensional)
                type_to_number -- dictionary mapping tree type to numerical label
    """
    fill = 0
    counter = 0
    while fill == 0:
        background, mask, free_area, fill = place_tree(distance, trees, background, mask, free_area, type_to_number)
        if fill == 0:
            counter += 1
        if verbose and counter % 50 == 0:
            print(f'{counter} trees placed.')

    print(f'\nForest has been filled. A total of {counter} additional trees have been placed.')
