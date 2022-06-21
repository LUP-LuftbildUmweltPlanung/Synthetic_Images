import os
import shutil
import warnings
import numpy as np
from pathlib import Path
from time import time
from itertools import repeat
from multiprocessing import cpu_count, Pool, current_process

import pandas as pd

import synth_forest as forest
from utils import save_image, unpack_results, store_results, get_files

# CONFIG START #
background_path = r'C:\DeepLearning_Local\+Projekte\SyntheticImageCreation\Daten\Background_cutouts\large_backgrounds\crops\New_adjusted\validation'
trees_path = r'C:\DeepLearning_Local\+Projekte\SyntheticImageCreation\Daten\tree_cutouts_2merged3\trees\+mit_bodo\combined\test_trees'
folder_name = 'testtestest'
##MIT BODO
label_dictionary = {'background': 0,
                    "BI": 1,
                    "BU": 2,
                    "EI": 3,
                    "ELA": 4,
                    "ER": 5,
                    "ES": 6,
                    "FI": 7,
                    "KI": 8,
                    "REI": 9,
                    "SHL": 10,
                    "SWL": 11,
                    "WLI": 12}
##OHNE BODO
# label_dictionary = {'background': 0,
#                     "BI": 1,
#                     "BU": 2,
#                     "EI": 3,
#                     "ELA": 4,
#                     "ER": 5,
#                     "FI": 6,
#                     "KI": 7,
#                     "SHL": 8,
#                     "SWL": 9}

area_per_pixel = 0.2 * 0.2
single_tree_distance = 10
shadow_percentile = 0.05

sparse_images = 5
single_cluster_images = 5
border_images = 5
dense_images = 5

path = r'C:\DeepLearning_Local\+Daten\+Synthetic_Images'

fill_with_same_tree = True
verbose = False
# CONFIG END #

if path is None:
    path = os.getcwd()
path = Path(path)


def sparse_image(*arg):
    """Creates an image containing sparsely placed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.verbose = arg[1]
    forest.fill_with_same_tree = arg[2]
    forest.type_to_number = arg[3]
    forest.number_to_type = {v: k for k, v in arg[3].items()}
    forest.trees = arg[4]
    idx = arg[0]

    forest.set_background(background_path, area_per_pixel, augment=True)
    forest.fill_with_trees(single_tree_distance)
    forest.finish_image()
    save_image(path / (folder_name + '/Sparse/sparse_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Sparse/sparse_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def single_cluster_image(*arg):
    """Creates an image containing a single, small, dense cluster and sparsely placed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.shadow_percentile = shadow_percentile
    forest.verbose = arg[1]
    forest.fill_with_same_tree = arg[2]
    forest.type_to_number = arg[3]
    forest.number_to_type = {v: k for k, v in arg[3].items()}
    forest.trees = arg[4]
    if fill_with_same_tree:
        forest.main_trees = pd.DataFrame({'tree_type': [arg[5][0]], 'file': [arg[5][1]]})
    idx = arg[0]

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


def border_image(*arg):
    """Creates an image containing a dense forest border and sparsely placed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.shadow_percentile = shadow_percentile
    forest.verbose = arg[1]
    forest.fill_with_same_tree = arg[2]
    forest.type_to_number = arg[3]
    forest.number_to_type = {v: k for k, v in arg[3].items()}
    forest.trees = arg[4]
    if fill_with_same_tree:
        forest.main_trees = pd.DataFrame({'tree_type': [arg[5][0]], 'file': [arg[5][1]]})
    idx = arg[0]

    forest.set_background(background_path, area_per_pixel, augment=True)
    forest.forest_edge()
    forest.fill_with_trees(single_tree_distance)
    forest.finish_image()
    save_image(path / (folder_name + '/Border/border_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Border/border_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def dense_image(*arg):
    """Creates an image containing densely packed trees.

                Keyword arguments:
                idx -- index of the current image to be used when storing mask and image
    """
    forest.shadow_percentile = shadow_percentile
    forest.verbose = arg[1]
    forest.fill_with_same_tree = arg[2]
    forest.type_to_number = arg[3]
    forest.number_to_type = {v: k for k, v in arg[3].items()}
    forest.trees = arg[4]
    if fill_with_same_tree:
        forest.main_trees = pd.DataFrame({'tree_type': [arg[5][0]], 'file': [arg[5][1]]})
    idx = arg[0]

    forest.set_background(background_path, area_per_pixel, augment=True)
    forest.dense_forest()
    forest.finish_image()
    save_image(path / (folder_name + '/Dense/dense_image_' + str(idx) + '.tif'), forest.background, forest.mask)
    trees, tree_types, tree_type_distribution, tree_type_distribution_no_back = forest.detailed_results()

    return forest.type_to_number, path / (folder_name + '/Dense/dense_image_' + str(idx) + '.tif'), \
           trees, tree_types, tree_type_distribution, tree_type_distribution_no_back


def create_images(type_to_number, tree_list, main_trees):
    """Creates the provided amount of images at the provided location."""
    cpus = cpu_count() - 1
    pool = Pool(processes=cpus, initializer=np.random.seed(current_process().pid))
    labels_and_paths = []

    if sparse_images:
        start = time()
        results = pool.starmap(sparse_image, zip(range(sparse_images), repeat(verbose), repeat(fill_with_same_tree),
                                                 repeat(type_to_number), repeat(tree_list)))
        results = unpack_results(results, sparse_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Sparse'))
        print(f'{sparse_images} sparse images have been created in {time() - start:.2f} seconds.\n')

    global single_cluster_images
    if single_cluster_images:
        start = time()
        if fill_with_same_tree:
            if single_cluster_images > len(main_trees):
                single_cluster_images = len(main_trees)
                warnings.warn(f"Not enough trees available. "
                              f"Only {single_cluster_images} single_cluster_images will be created.")
            else:
                main_trees = main_trees.sample(frac=1)
            results = pool.starmap(single_cluster_image, zip(range(single_cluster_images),
                                                             repeat(verbose), repeat(fill_with_same_tree),
                                                             repeat(type_to_number), repeat(tree_list),
                                                             main_trees.head(single_cluster_images).values.tolist()))
        else:
            results = pool.starmap(single_cluster_image, zip(range(single_cluster_images),
                                                             repeat(verbose), repeat(fill_with_same_tree),
                                                             repeat(type_to_number), repeat(tree_list)))
        results = unpack_results(results, single_cluster_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Single_cluster'))
        main_trees = tree_list
        print(f'{single_cluster_images} single cluster images have been created in {time() - start:.2f} seconds.\n')

    global border_images
    if border_images:
        start = time()
        if fill_with_same_tree:
            if border_images > len(main_trees):
                border_images = len(main_trees)
                warnings.warn(f"Not enough trees available. "
                              f"Only {border_images} border_images will be created.")
            else:
                main_trees = main_trees.sample(frac=1)
            results = pool.starmap(border_image, zip(range(border_images),
                                                     repeat(verbose), repeat(fill_with_same_tree),
                                                     repeat(type_to_number), repeat(tree_list),
                                                     main_trees.head(border_images).values.tolist()))
        else:
            results = pool.starmap(border_image, zip(range(border_images),
                                                     repeat(verbose), repeat(fill_with_same_tree),
                                                     repeat(type_to_number), repeat(tree_list)))
        results = unpack_results(results, border_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Border'))
        main_trees = tree_list
        print(f'{border_images} border images have been created in {time() - start:.2f} seconds.\n')

    global dense_images
    if dense_images:
        start = time()
        if fill_with_same_tree:
            if dense_images > len(main_trees):
                dense_images = len(main_trees)
                warnings.warn(f"Not enough trees available. "
                              f"Only {dense_images} dense_images will be created.")
            else:
                main_trees = main_trees.sample(frac=1)
            results = pool.starmap(dense_image, zip(range(dense_images),
                                                    repeat(verbose), repeat(fill_with_same_tree),
                                                    repeat(type_to_number), repeat(tree_list),
                                                    main_trees.head(dense_images).values.tolist()))
        else:
            results = pool.starmap(dense_image, zip(range(dense_images),
                                                    repeat(verbose), repeat(fill_with_same_tree),
                                                    repeat(type_to_number), repeat(tree_list)))
        results = unpack_results(results, dense_images)
        labels_and_paths += results[0]
        store_results(results[1:], path=path / (folder_name + '/Dense'))
        main_trees = tree_list
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
    forest.get_trees(trees_path)
    type_to_number = forest.type_to_number
    number_to_type = forest.number_to_type
    tree_list = forest.trees
    main_trees = forest.main_trees

    warn = False
    if label_dictionary is not None:
        for label in type_to_number.keys():
            if label not in label_dictionary.keys():
                class_value = len(label_dictionary.keys())
                label_dictionary[label] = class_value
                warn = True

        type_to_number = label_dictionary
        number_to_type = dict((v, k) for k, v in type_to_number.items())

        if warn:
            warnings.warn(
                f"Provided label dictionary did not cover all tree types found. It was updated to: {type_to_number}.")

    create_images(type_to_number, tree_list, main_trees)
    prepare_files_for_unet(destination=path / folder_name / "Unet_Format")

# notes:    - update documentation
#           - add warnings: warnings.warn("Warning......message")
#           - add verbosity levels (None, basic, relevant, detailed)
#
# functions: - option to disable multiprocessing
#
# issues:   - clusters are placed at random, without considering free area
#           - percentage of background in overview sometimes over 100%
#           - blue/pink fields
