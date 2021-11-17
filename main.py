import synth_forest as forest
from utils import save_image

# CONFIG START #
background_file = r'Example_Tree_Data/Background/test_background_8bit.tif'
trees_path = r'Example_Tree_Data/8bit'
area_per_pixel = 0.2 * 0.2
save = False
verbose = True
# CONFIG END #

forest.verbose = verbose

background, mask, free_area, height_mask = forest.set_background(background_file, area_per_pixel, augment=True)
limits = background.shape
trees, type_to_number, number_to_type = forest.get_trees(trees_path)

cluster_area = 10000
single_tree_distance = 10
forest.place_cluster(cluster_area, trees, background, mask, free_area, height_mask, type_to_number)
forest.fill_with_trees(single_tree_distance, trees, background, mask, free_area, height_mask, type_to_number)
forest.visualize(background, mask, height_mask)
forest.detailed_results(mask, number_to_type)

if save:
    save_image(r'Example_Tree_Data/Results/test_background_8bit_synth.tif', background, mask)

# notes:    - cut out better trees
#           - randomize distance for single trees further
#           - add warnings: warnings.warn("Warning......message")
#           - add verbosity levels (None, basic, relevant, detailed)
#
# functions: ---
#
# issues:   - distance around tree only effects center pixel of new trees --> new large trees overlap anyway
#           - clusters are placed at random, without considering free area
