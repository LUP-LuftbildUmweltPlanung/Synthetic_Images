import os
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'

area_per_pixel = 0.2 * 0.2
single_tree_distance = 10

sparse_images = 5
single_cluster_images = 5
border_images = 5
dense_images = 5

path = None

verbose = False
# CONFIG END #

forest.verbose = verbose
if path is None:
    path = os.getcwd()

labels = []
paths = []


def sparse_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    forest.fill_with_trees(single_tree_distance)
    save_image(path + '/Images/Sparse/sparse_image_' + str(idx) + '.tif', forest.background, forest.mask)
    paths.append(path + '/Images/Sparse/sparse_image_' + str(idx) + '.tif')
    if forest.type_to_number not in labels:
        labels.append(forest.type_to_number)


def single_cluster_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    max_area = forest.background.size * area_per_pixel
    area = np.random.choice(np.arange(int(max_area / 10), max_area))
    forest.place_cluster(area)
    forest.fill_with_trees(single_tree_distance)
    save_image(path + '/Images/Single_cluster/single_cluster_image_' + str(idx) + '.tif', forest.background,
               forest.mask)
    paths.append(path + '/Images/Sparse/sparse_image_' + str(idx) + '.tif')
    if forest.type_to_number not in labels:
        labels.append(forest.type_to_number)


def border_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    forest.forest_edge()
    forest.fill_with_trees(single_tree_distance)
    save_image(path + '/Images/Border/border_image_' + str(idx) + '.tif', forest.background, forest.mask)
    paths.append(path + '/Images/Border/border_image_' + str(idx) + '.tif')
    if forest.type_to_number not in labels:
        labels.append(forest.type_to_number)


def dense_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    forest.dense_forest()
    save_image(path + '/Images/Dense/dense_image_' + str(idx) + '.tif', forest.background, forest.mask)
    paths.append(path + '/Images/Sparse/sparse_image_' + str(idx) + '.tif')
    if forest.type_to_number not in labels:
        labels.append(forest.type_to_number)


def create_images():
    cpus = cpu_count() - 1
    pool = Pool(processes=cpus)

    pool.map(sparse_image, range(sparse_images))
    pool.map(single_cluster_image, range(single_cluster_images))
    pool.map(border_image, range(border_images))
    pool.map(dense_image, range(dense_images))

    with open(path + 'labels.txt', 'w') as file:
        for l in labels:
            file.write(l)
            file.write('\n')

        for p in paths:
            file.write(p)
            file.write('\n')


if __name__ == '__main__':
    create_images()

# notes:    - update documentation
#           - cut out better trees
#           - add warnings: warnings.warn("Warning......message")
#           - add verbosity levels (None, basic, relevant, detailed)
#
# functions: - overarching config --> creation of hundreds of pictures
#
# issues:   - clusters are placed at random, without considering free area
#           - slow performance due to buffer calculation (in place_tree function) --> multi-processing?
