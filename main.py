import matplotlib.pyplot as plt

import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'
number_of_trees = 0
distance = 10
area_per_pixel = 0.2 * 0.2
save = False
# CONFIG END #

background, mask, free_area, height_mask = forest.set_background(background_file, area_per_pixel, augment=True)
limits = background.shape
trees, type_to_number, number_to_type = forest.get_trees(trees_path)

forest.place_cluster(10000, trees, background, mask, free_area, height_mask, type_to_number)

forest.fill_with_trees(distance, trees, background, mask, free_area, height_mask, type_to_number)

print(f'\nTotal number of trees placed: {forest.tree_counter}.')
print(f'Distribution of tree types: {forest.tree_type_counter}.')

print(f'Percentage: {forest.tree_type_distribution(mask, number_to_type)}.')
print(f'Percentage without background: {forest.tree_type_distribution(mask, number_to_type, background=False)}.')

plt.imshow(background)
plt.title('Synthetic image')
plt.show()

plt.imshow(mask, cmap='hot')
plt.title('Image mask')
plt.show()

plt.imshow(height_mask, cmap='Greys_r')
plt.title('Height map')
plt.show()

if save:
    save_image(r'Example_Tree_Data/Results/test_background_8bit_synth.tif', background, mask)

# notes:    - cut out better trees
#           - set distance to random
#           - update documentation
#
# functions: ---
#
# issues:   - distance around tree only effects center pixel of new trees --> new large trees overlap anyway
#           - clusters are placed at random, without considering free area
