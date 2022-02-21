import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

directory = r"C:\DeepLearning_Local\+Daten\+Waldmasken\fuer_synthetic_images\tree_cutouts2\trees_radiometric_corrected\trees_8bit"

directory = Path(directory)
ori_dir = os.getcwd()
os.chdir(directory)
files = [directory / file for file in glob.glob('*.tif')]
os.chdir(ori_dir)

tree_data = {'number': [], 'ba': [], 'bagr': [], 'path': []}
for f in files:
    single_tree = str(f).rsplit('\\', 1)[-1]
    tree_data['number'].append(single_tree.rsplit('_', 1)[-1].rsplit('.', 1)[0])
    tree_data['ba'].append(single_tree.split('_', 1)[0])
    tree_data['bagr'].append(single_tree.split('_', 1)[-1].split('_', 1)[0])
    tree_data['path'].append(f)

tree_data = pd.DataFrame(tree_data)

ba_list = list(set(tree_data['ba'].to_list()))

train_trees = []
test_trees = []

for ba in ba_list:
    path_list = tree_data.loc[tree_data['ba'] == ba]['path'].tolist()
    np.random.shuffle(path_list)

    train_length = int(len(path_list) * 0.7)

    train_trees.extend(path_list[:train_length])
    test_trees.extend(path_list[train_length:])

Path(directory / 'train_trees').mkdir(parents=True, exist_ok=True)
Path(directory / 'test_trees').mkdir(parents=True, exist_ok=True)

train_trees_16bit = []
test_trees_16bit = []

for tree in train_trees:
    os.rename(tree, directory / ('train_trees/' + str(tree).rsplit('\\', 1)[-1]))
    train_trees_16bit.append(str(tree).rsplit('_', 1)[-1].split('.', 1)[0])

for tree in test_trees:
    os.rename(tree, directory / ('test_trees/' + str(tree).rsplit('\\', 1)[-1]))
    test_trees_16bit.append(str(tree).rsplit('_', 1)[-1].split('.', 1)[0])

print('All 8bit images moved successfully.')

directory = r"C:\DeepLearning_Local\+Daten\+Waldmasken\fuer_synthetic_images\tree_cutouts2\trees_radiometric_corrected\trees_16bit"

directory = Path(directory)
ori_dir = os.getcwd()
os.chdir(directory)
files = [directory / file for file in glob.glob('*')]
os.chdir(ori_dir)

tree_data_16bit = {'number': [], 'path': []}

for f in files:
    if '.' in str(f):
        number = str(f).rsplit('_', 1)[-1].split('.', 1)[0]
        tree_data_16bit['number'].append(number)
        tree_data_16bit['path'].append(f)

tree_data_16bit = pd.DataFrame(tree_data_16bit)

train_tree_16bit_path = []
for number in train_trees_16bit:
    train_tree_16bit_path.extend(tree_data_16bit.loc[tree_data_16bit['number'] == number]['path'].tolist())

test_tree_16bit_path = []
for number in test_trees_16bit:
    test_tree_16bit_path.extend(tree_data_16bit.loc[tree_data_16bit['number'] == number]['path'].tolist())

Path(directory / 'train_trees').mkdir(parents=True, exist_ok=True)
Path(directory / 'test_trees').mkdir(parents=True, exist_ok=True)

for tree in train_tree_16bit_path:
    os.rename(tree, directory / ('train_trees/' + str(tree).rsplit('\\', 1)[-1]))

for tree in test_tree_16bit_path:
    os.rename(tree, directory / ('test_trees/' + str(tree).rsplit('\\', 1)[-1]))

print('All 16bit images moved successfully.')
