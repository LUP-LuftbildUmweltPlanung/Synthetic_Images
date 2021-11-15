import matplotlib.pyplot as plt

import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'
number_of_trees = 10
distance = 50
# CONFIG END #

background, mask, free_area = forest.set_background(background_file)
limits = background.shape
trees, type_to_number, number_to_type = forest.get_trees(trees_path)

for i in range(number_of_trees):
    background, mask, free_area, full = forest.place_tree(distance, trees, background, mask, free_area, type_to_number)
    if full == 1:
        break

distance = 10
# forest.fill_with_trees(distance, trees, background, mask, free_area, type_to_number)

print(f'\nTotal number of trees placed: {forest.tree_counter}.')
print(f'\nDistribution of tree types: {forest.tree_type_counter}.')

plt.imshow(background)
plt.show()

plt.imshow(mask, cmap='hot')
plt.show()

# plt.imshow(free_area * 255, cmap='gray', vmin=0, vmax=255)
# plt.show()

# save_image(r'Example_Tree_Data/8bit/test_background_8bit_synth.tif', background)

# notes:    add (basic) tree augmentations, move useful parts of "place_tree" to utils
# functions:    add tree cluster (trees need a smaller of limits area, so they are not inside each other)
#                   param: tree-distance, tree-amount
#
# issues:   if tree contains black pixels (value=0), mask contains holes accordingly
