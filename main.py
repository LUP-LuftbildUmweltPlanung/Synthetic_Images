import os
import numpy as np
from tqdm import tqdm

import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'

area_per_pixel = 0.2 * 0.2
single_tree_distance = 10

sparse = 3
single_cluster = 3
border = 3
dense = 3

path = None

verbose = False
# CONFIG END #

forest.verbose = verbose


def create_images(path):
    if path is None:
        path = os.getcwd()
    forest.get_trees(trees_path)
    for s_idx in tqdm(range(sparse)):
        forest.set_background(background_file, area_per_pixel, augment=True)
        forest.fill_with_trees(single_tree_distance)
        save_image(path + '/Images/Sparse/sparse_image_' + str(s_idx) + '.tif', forest.background, forest.mask)

    for sc_idx in tqdm(range(single_cluster)):
        forest.set_background(background_file, area_per_pixel, augment=True)
        max_area = forest.background.size * area_per_pixel
        area = np.random.choice(np.arange(int(max_area / 10), max_area))
        forest.place_cluster(area)
        forest.fill_with_trees(single_tree_distance)
        save_image(path + '/Images/Single_cluster/single_cluster_image_' + str(sc_idx) + '.tif', forest.background, forest.mask)

    for b_idx in tqdm(range(border)):
        forest.set_background(background_file, area_per_pixel, augment=True)
        forest.forest_edge()
        forest.fill_with_trees(single_tree_distance)
        save_image(path + '/Images/Border/border_image_' + str(b_idx) + '.tif', forest.background, forest.mask)

    for d_idx in tqdm(range(dense)):
        forest.set_background(background_file, area_per_pixel, augment=True)
        forest.dense_forest()
        save_image(path + '/Images/Dense/dense_image_' + str(d_idx) + '.tif', forest.background, forest.mask)


if __name__ == '__main__':
    create_images(path)

# notes:    - update documentation
#           - cut out better trees
#           - add warnings: warnings.warn("Warning......message")
#           - add verbosity levels (None, basic, relevant, detailed)
#
# functions: - overarching config --> creation of hundreds of pictures
#
# issues:   - clusters are placed at random, without considering free area
#           - slow performance due to buffer calculation (in place_tree function) --> multi-processing?
