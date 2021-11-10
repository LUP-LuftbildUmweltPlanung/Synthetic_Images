import cv2
import matplotlib.pyplot as plt
import numpy as np

trees = {0: 'ELA', 1: 'GES', 2: 'RER', 3: 'ROB', 4: 'SEI'}  # defines files, needs to be replaced with glob
images = {}

for idx in trees.keys():
    # currently stores all images in memory
    # should be upgraded to tore image paths instead
    path = r'Example_Tree_Data/8bit/' + trees[idx] + '_8bit.tif'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # loads 4-band image
    images[idx] = img[:, :, :]  # stores 3-band image in image dict

path = r'Example_Tree_Data/8bit/test_background_8bit.tif'
background = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, :]  # loads 3-band version of background

limits = background.shape

number_of_trees = 100

for i in range(number_of_trees):
    # loop to add trees
    x = np.random.choice(limits[0])  # x_postion for tree
    y = np.random.choice(limits[1])  # y_position for tree

    img_idx = np.random.choice(len(images))  # random tree
    img = images[img_idx]
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

cv2.imwrite(r'Example_Tree_Data/8bit/test_background_8bit_synth.tif', background)

# notes:    add (basic) tree augmentations, add tree type counter, add min_distance for trees
#           update image storage to image path storage
