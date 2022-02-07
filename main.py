import os
import shutil
import numpy as np
from pathlib import Path
from time import time
from multiprocessing import cpu_count, Pool, current_process

import synth_forest as forest
from utils import save_image, unpack_results, store_results, get_files

# CONFIG START #
background_path = r'C:\DeepLearning_Local\+Daten\+Waldmasken\fuer_synthetic_images\Background_cutouts\backgrounds\40cm\background_8bit\Train'
trees_path = r'C:\DeepLearning_Local\+Daten\+Waldmasken\fuer_synthetic_images\tree_cutouts2\trees\trees_8bit\train_trees'
folder_name = 'test_final'

label_dictionary = {'background': 0,
                    "BAH": 1,
                    "BI": 2,
                    "BU": 3,
                    "EI": 4,
                    "ELA": 5,
                    "ER": 6,
                    "FI": 7,
                    "GES": 8,
                    "HBU": 9,
                    "KI": 10,
                    "REI": 11,
                    "SHL": 12,
                    "SKI": 13,
                    "SWL": 14,
                    "WLI": 15}

area_per_pixel = 0.2 * 0.2
single_tree_distance = 10

sparse_images = 10
single_cluster_images = 10
border_images = 10
dense_images = 10

path = r'C:\DeepLearning_Local\+Daten\+Synthetic_Images'
unet_format = True

verbose = False
# CONFIG END #

forest.verbose = verbose
if path is None:
    path = os.getcwd()
path = Path(path)

forest.get_trees(trees_path)
type_to_number = forest.type_to_number
number_to_type = forest.number_to_type
tree_list = forest.trees

if label_dictionary is not None:
    for label in type_to_number.keys():
        if label not in label_dictionary.keys():
            class_value = len(label_dictionary.keys())
            label_dictionary[label] = class_value

    type_to_number = label_dictionary
    number_to_type = dict((v, k) for k, v in type_to_number.items())


def sparse_image(idx):
    """Creates an image containing sparsely placed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.type_to_number = type_to_number
    forest.number_to_type = number_to_type
    forest.trees = tree_list

    forest.set_background(background_path, area_per_pixel, augment=True)
    forest.fill_with_trees(single_tree_distance)
    forest.finish_image()
    save_image(path / (folder_name + '/Sparse/sparse_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Sparse/sparse_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def single_cluster_image(idx):
    """Creates an image containing a single, small, dense cluster and sparsely placed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.type_to_number = type_to_number
    forest.number_to_type = number_to_type
    forest.trees = tree_list

    forest.set_background(background_path, area_per_pixel, augment=True)
    max_area = forest.background.shape[0] * forest.background.shape[1] * area_per_pixel
    area = np.random.choice(np.arange(int(max_area / 10), max_area))
    forest.place_cluster(area)
    forest.fill_with_trees(single_tree_distance)
    forest.finish_image()
    save_image(path / (folder_name + '/Single_cluster/single_cluster_image_' + str(idx) + '.tif'),
               forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Single_cluster/single_cluster_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def border_image(idx):
    """Creates an image containing a dense forest border and sparsely placed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.type_to_number = type_to_number
    forest.number_to_type = number_to_type
    forest.trees = tree_list

    forest.set_background(background_path, area_per_pixel, augment=True)
    forest.forest_edge()
    forest.fill_with_trees(single_tree_distance)
    forest.finish_image()
    save_image(path / (folder_name + '/Border/border_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Border/border_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def dense_image(idx):
    """Creates an image containing densely packed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.type_to_number = type_to_number
    forest.number_to_type = number_to_type
    forest.trees = tree_list

    forest.set_background(background_path, area_per_pixel, augment=True)
    forest.dense_forest()
    forest.finish_image()
    save_image(path / (folder_name + '/Dense/dense_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Dense/dense_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def create_images():
    """Creates the provided amount of images at the provided location."""
    cpus = cpu_count() - 1
    pool = Pool(processes=cpus, initializer=np.random.seed(current_process().pid))
    labels_and_paths = []

    if sparse_images:
        start = time()
        results = pool.map(sparse_image, range(sparse_images))
        results = unpack_results(results, sparse_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Sparse'))
        print(f'{sparse_images} sparse images have been created in {time() - start:.2f} seconds.\n')

    if single_cluster_images:
        start = time()
        results = pool.map(single_cluster_image, range(single_cluster_images))
        results = unpack_results(results, single_cluster_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Single_cluster'))
        print(f'{single_cluster_images} single cluster images have been created in {time() - start:.2f} seconds.\n')

    if border_images:
        start = time()
        results = pool.map(border_image, range(border_images))
        results = unpack_results(results, border_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Border'))
        print(f'{border_images} border images have been created in {time() - start:.2f} seconds.\n')

    if dense_images:
        start = time()
        results = pool.map(dense_image, range(dense_images))
        results = unpack_results(results, dense_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Dense'))
        print(f'{dense_images} dense images have been created in {time() - start:.2f} seconds.\n')

    with open(path / (folder_name + '/labels.txt'), 'w') as file:
        file.write(str(type_to_number))
        file.write('\n\n')
        for l_p in labels_and_paths:
            l, p = l_p

            file.write(str(p))
            file.write('\n')


def prepare_files_for_unet(destination):
    """Copies the files from the provided path (in the CONFIG) to the destination folder in a UNET required structure.

                Keyword arguments:
                destination -- folder to store the files in a UNET required structure in
    """
    destination = Path(destination)
    source = path / folder_name
    sources = [p.path for p in os.scandir(str(source)) if p.is_dir()]
    Path(destination / 'mask_tiles').mkdir(parents=True, exist_ok=True)
    Path(destination / 'img_tiles').mkdir(parents=True, exist_ok=True)
    for s in sources:
        files = get_files(s, 'tif')
        for f in files:
            if str(f).rsplit('_', 1)[-1] == 'mask.tif':
                dest = destination / 'mask_tiles'
            else:
                dest = destination / 'img_tiles'

            try:
                shutil.copy(f, dest)

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")

            if str(f).rsplit('_', 1)[-1] == 'mask.tif':
                os.rename(dest / str(f).rsplit('\\')[-1],
                          dest / (str(f).rsplit('\\', 1)[-1].rsplit('_', 1)[0] + '.tif'))

    print("Copied all files successfully.")


if __name__ == '__main__':
    create_images()
    prepare_files_for_unet(destination=path / folder_name / "Unet_Format")

# notes:    - update documentation
#           - add warnings: warnings.warn("Warning......message")
#           - add verbosity levels (None, basic, relevant, detailed)
#
# functions: - option to disable multiprocessing
#
# issues:   - clusters are placed at random, without considering free area
#           - mulitprocessing mixes the results. As results are calculated per image class,
#           it should not cause an issue, but may cause other problems or even mistakes in the distribution
#           (error not fully understood)
#           - percentage of background in overview sometimes over 100%
