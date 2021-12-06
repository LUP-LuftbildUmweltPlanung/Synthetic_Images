import os
import numpy as np
from pathlib import Path
from time import time
from multiprocessing import cpu_count, Pool

import synth_forest as forest
from utils import save_image, unpack_results, store_results

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'C:\DeepLearning_Local\+Daten\+Waldmasken\Tree_cutouts\trees_8bit'
folder_name = 'Test_test_test'

area_per_pixel = 0.2 * 0.2
single_tree_distance = 10

sparse_images = 10
single_cluster_images = 10
border_images = 10
dense_images = 10

path = 'C:\DeepLearning_Local\+Daten\+Synthetic_Images'

verbose = False
# CONFIG END #

forest.verbose = verbose
if path is None:
    path = os.getcwd()
path = Path(path)


def sparse_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    forest.fill_with_trees(single_tree_distance)
    save_image(path / (folder_name + '/Sparse/sparse_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Sparse/sparse_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def single_cluster_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    max_area = forest.background.size * area_per_pixel
    area = np.random.choice(np.arange(int(max_area / 10), max_area))
    forest.place_cluster(area)
    forest.fill_with_trees(single_tree_distance)
    save_image(path / (folder_name + '/Single_cluster/single_cluster_image_' + str(idx) + '.tif'),
               forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Single_cluster/single_cluster_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def border_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    forest.forest_edge()
    forest.fill_with_trees(single_tree_distance)
    save_image(path / (folder_name + '/Border/border_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Border/border_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def dense_image(idx):
    forest.get_trees(trees_path)
    forest.set_background(background_file, area_per_pixel, augment=True)
    forest.dense_forest()
    save_image(path / (folder_name + '/Dense/dense_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Dense/dense_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def create_images():
    cpus = cpu_count() - 1
    pool = Pool(processes=cpus)
    labels_and_paths = []

    start = time()
    results = pool.map(sparse_image, range(sparse_images))
    results = unpack_results(results, sparse_images)
    labels_and_paths += results[0]
    store_results(results[1:], path=path / (folder_name + '/Sparse'))
    print(f'{sparse_images} sparse images have been created in {time() - start:.2f} seconds.\n')

    start = time()
    results = pool.map(single_cluster_image, range(single_cluster_images))
    results = unpack_results(results, single_cluster_images)
    labels_and_paths += results[0]
    store_results(results[1:], path=path / (folder_name + '/Single_cluster'))
    print(f'{single_cluster_images} single cluster images have been created in {time() - start:.2f} seconds.\n')

    start = time()
    results = pool.map(border_image, range(border_images))
    results = unpack_results(results, border_images)
    labels_and_paths += results[0]
    store_results(results[1:], path=path / (folder_name + '/Border'))
    print(f'{border_images} border images have been created in {time() - start:.2f} seconds.\n')

    start = time()
    results = pool.map(dense_image, range(dense_images))
    results = unpack_results(results, dense_images)
    labels_and_paths += results[0]
    store_results(results[1:], path=path / (folder_name + '/Dense'))
    print(f'{dense_images} dense images have been created in {time() - start:.2f} seconds.\n')

    with open(path / (folder_name + '/labels.txt'), 'w') as file:
        labels = []
        for l_p in labels_and_paths:
            l, p = l_p
            if l not in labels:
                labels.append(l)
                file.write(str(l))
                file.write('\n\n')

            file.write(str(p))
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
