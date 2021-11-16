from utils import *
from bezier_shape import random_shape

tree_counter = 0
tree_type_counter = {}


def set_background(file_path, reset=True, augment=False, bands=None):
    """Loads a image files as background. Unless specified, also provides a fresh label mask and blocked_area mask.

        Keyword arguments:
        file_path -- path to the image file
        reset -- resets mask and blocked_area (default True)
        bands -- sets bands to specified value, leaves as is if None (default None)
    """
    background = load_image(file_path, bands)
    if augment:
        background = background_augmentation(background)
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


def place_tree(distance, trees, background, mask, free_area, type_to_number, augment=True, fill=False):
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

    x, y = random_position(free_area)  # selects a random position in the image

    tree, tree_type = random_tree(trees, augment)  # selects a tree at random from a list of trees
    tree_label = type_to_number[tree_type]  # converts tree_type to label

    boundaries = background.shape

    x_area, y_area, tree = set_area(x, y, tree, boundaries)  # sets image area, crops if necessary

    background, mask = place_in_background(tree, tree_label, x_area, y_area, background, mask)

    if fill:
        shape_type = 'fill'
        distance = int(np.mean(tree.shape[:2])/1.4)
    else:
        shape_type = 'single_tree'

    block_shape = random_shape(distance*2, shape_type)
    x_block_area, y_block_area, block_shape = set_area(x, y, block_shape, boundaries)

    free_area[x_block_area[0]:x_block_area[1], y_block_area[0]:y_block_area[1]] *= block_shape == 0  # sets blocked area

    global tree_counter, tree_type_counter
    tree_counter += 1
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
        background, mask, free_area, fill = \
            place_tree(distance, trees, background, mask, free_area, type_to_number, fill=True)
        if fill == 0:
            counter += 1
        if verbose and counter % 50 == 0:
            print(f'{counter} trees placed.')

    print(f'\nForest has been filled. A total of {counter} additional trees have been placed.')
