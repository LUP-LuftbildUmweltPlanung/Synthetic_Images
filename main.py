import matplotlib.pyplot as plt

import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'
number_of_trees = 0
distance = 50
pixel_area = 0.2 * 0.2
save = False
# CONFIG END #

background, mask, free_area = forest.set_background(background_file, pixel_area)
limits = background.shape
trees, type_to_number, number_to_type = forest.get_trees(trees_path)

for i in range(number_of_trees):
    full = forest.place_tree(distance, trees, background, mask, free_area, type_to_number)
    if full:
        break

# distance = 20
# forest.fill_with_trees(distance, trees, background, mask, free_area, type_to_number)
forest.place_cluster(10000, trees, background, mask, free_area, type_to_number)

print(f'\nTotal number of trees placed: {forest.tree_counter}.')
print(f'Distribution of tree types: {forest.tree_type_counter}.')

print(f'Percentage: {forest.tree_type_distribution(mask, number_to_type, background=True)}.')
print(f'Percentage without background: {forest.tree_type_distribution(mask, number_to_type)}.')

plt.imshow(background)
plt.show()

plt.imshow(mask, cmap='hot')
plt.show()

if save:
    save_image(r'Example_Tree_Data/Results/test_background_8bit_synth.tif', background, mask)

# notes:    - cut out better trees
#           - add height to cluster function
#           - set distance to random
#
# functions: ...
#
# issues:   - if tree contains black pixels (value=0), mask contains holes accordingly
#           - distance around tree only effects center pixel of new trees --> new large trees overlap anyway
