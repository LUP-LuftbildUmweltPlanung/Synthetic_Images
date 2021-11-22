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

forest.set_background(background_file, area_per_pixel, augment=True)
forest.get_trees(trees_path)

cluster_area = 1000
single_tree_distance = 10
forest.place_cluster(cluster_area)
forest.fill_with_trees(single_tree_distance)
forest.visualize()
forest.detailed_results()

if save:
    save_image(r'Example_Tree_Data/Results/test_background_8bit_synth.tif', forest.background, mask)

# notes:    - update documentation
#           - cut out better trees
#           - randomize distance for single trees further
#           - add warnings: warnings.warn("Warning......message")
#           - add verbosity levels (None, basic, relevant, detailed)
#
# functions: - overarching config --> creation of hundreds of pictures
#
# issues:   - clusters are placed at random, without considering free area
