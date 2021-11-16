import matplotlib.pyplot as plt

import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'
number_of_trees = 10
distance = 50
save = True
# CONFIG END #

background, mask, free_area = forest.set_background(background_file)
limits = background.shape
trees, type_to_number, number_to_type = forest.get_trees(trees_path)

for i in range(number_of_trees):
    background, mask, free_area, full = forest.place_tree(distance, trees, background, mask, free_area, type_to_number)
    # plt.imshow(free_area, cmap='hot')
    # plt.show()
    if full == 1:
        break

distance = 20
forest.fill_with_trees(distance, trees, background, mask, free_area, type_to_number)
# forest.place_cluster(trees, background, mask, free_area, type_to_number, tree_amount=500)

print(f'\nTotal number of trees placed: {forest.tree_counter}.')
print(f'Distribution of tree types: {forest.tree_type_counter}.')

plt.imshow(background)
plt.show()

plt.imshow(mask, cmap='hot')
plt.show()

if save:
    save_image(r'Example_Tree_Data/Results/test_background_8bit_synth.tif', background, mask)

# notes:    - cut out better trees
#           - add height to cluster function
#           - count pixel per type
#           - set distance to random
# functions:- new cluster function
#               - measure random shape size
#               - resize to match input cluster size
#               - fill with variable dist and random height trees
#
# issues:   - if tree contains black pixels (value=0), mask contains holes accordingly
