import matplotlib.pyplot as plt
import numpy as np

import synth_forest as forest
from utils import load_image, save_image

# CONFIG #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'
number_of_trees = 100

background, mask, blocked_area = forest.set_background(background_file)
limits = background.shape
trees = forest.get_trees(trees_path)

for i in range(number_of_trees):
    # loop to add trees
    x = np.random.choice(limits[0])  # x_position for tree
    y = np.random.choice(limits[1])  # y_position for tree

    img_idx = np.random.choice(len(trees))  # random tree
    img = load_image(trees[img_idx][0])

    img_shape = img.shape  # gets image shape
    img_center = (int(img_shape[0] / 2), int(img_shape[1] / 2))  # gets image center

    x_range = [x - img_center[0],
               x - img_center[0] + img_shape[0]]  # gets x_range according do image shape and x position
    y_range = [y - img_center[1],
               y - img_center[1] + img_shape[1]]  # gets y_range according do image shape and y position

    if x_range[0] < 0:  # checks, if image is out of left bound
        overlap = -1 * x_range[0]
        img = img[overlap:]
    elif x_range[1] > limits[0]:  # checks, if image is out of right bound
        overlap = x_range[1] - limits[0]
        img = img[:-overlap]

    if y_range[0] < 0:  # checks, if image is out of upper bound
        overlap = -1 * y_range[0]
        img = img[:, overlap:]
    elif y_range[1] > limits[1]:  # checks, if image is out of lower bound
        overlap = y_range[1] - limits[1]
        img = img[:, :-overlap]

    img_mask = img == 0  # mask to only remove tree part of image

    x_range = np.clip(x_range, 0, limits[0])  # clip x_range to image bounds
    y_range = np.clip(y_range, 0, limits[1])  # clip y_range to image bounds

    background[x_range[0]:x_range[1], y_range[0]:y_range[1]] *= img_mask  # empty tree area in background
    background[x_range[0]:x_range[1], y_range[0]:y_range[1]] += img  # add tree into freshly deleted area

plt.imshow(background)
plt.show()

# save_image(r'Example_Tree_Data/8bit/test_background_8bit_synth.tif', background)

# notes:    add (basic) tree augmentations, add tree type counter
#           update image storage to image path storage
# functions:    add single tree (needs a somewhat randomized area around it, where no trees can be placed)
#                   param: tree-area
#               add tree cluster (trees need a smaller of limits area, so they are not inside each other)
#                   param: tree-distance, tree-amount
#               add fill function (places trees until all areas outside of tree-area contain a tree)
