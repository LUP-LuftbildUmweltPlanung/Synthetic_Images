from utils import *


def place_cluster(trees, background, mask, free_area, type_to_number, tree_amount=None, cluster_area=None):
    """Creates a cluster of trees at a random point, updates the image mask (and free area not yet) respectively.
    Based on either tree-amount or cluster-area.

                Keyword arguments:
                trees -- list containing tuples of type (tree_image_path, tree_type)
                background -- array containing the image (N-dimensional)
                mask -- array containing the image mask (1-dimensional)
                free_area -- array containing 1 where trees can be placed (1-dimensional)
                type_to_number -- dictionary mapping tree type to numerical label
                tree_amount -- Amount of trees in cluster (default = None)
                cluster_area -- Size of the area of the created cluster (default = None)
        """
    if tree_amount is None and cluster_area is None:
        print('\nNo tree amount or cluster area has been defined. The tree cluster was not placed.')

    elif tree_amount is not None and cluster_area is not None:
        print('\nBoth tree amount and cluster area have been defined. '
              'The cluster area will be ignored, and the tree amount while be used.')

    if np.sum(free_area) == 0:
        print('\nImage does not contain any free area anymore. The tree cluster was not placed.')
        return background, mask, free_area, 1

    x, y = random_position(free_area)  # selects a random position in the image

    tree, tree_type = random_tree(trees, augment=True)  # selects a tree at random from a list of trees
    tree_label = type_to_number[tree_type]  # converts tree_type to label

    boundaries = background.shape

    x_area, y_area, tree = set_area(x, y, tree, boundaries)  # sets image area, crops if necessary

    background, mask = place_in_background(tree, tree_label, x_area, y_area, background, mask)  # places image

    global tree_counter
    tree_counter += 1

    global tree_type_counter
    if tree_type not in tree_type_counter.keys():
        tree_type_counter[tree_type] = 0
    tree_type_counter[tree_type] += 1

    if tree_amount is not None:
        area = np.concatenate([x_area, y_area])
        cluster_mask = tree[:, :, 0] != 0
        for i in range(tree_amount - 1):
            tree, tree_type = random_tree(trees, augment=False)  # selects a tree at random from a list of trees
            tree_label = type_to_number[tree_type]  # converts tree_type to label

            x, y = contact_position(area, cluster_mask, tree)  # calculates the position required for the tree to
            # contact the cluster at a random position

            x_area, y_area, tree = set_area(x, y, tree, boundaries)  # sets image area, crops if necessary

            background, mask = place_in_background(tree, tree_label, x_area, y_area, background, mask)  # places image

            area = np.array([np.min([area[0], x_area[0]]), np.max([area[1], x_area[1]]),
                             np.min([area[2], y_area[0]]), np.max([area[3], y_area[1]])])  # updates cluster area

            cluster_mask = mask[area[0]:area[1], area[2]:area[3]] != 0  # updates cluster mask

            tree_counter += 1
            if tree_type not in tree_type_counter.keys():
                tree_type_counter[tree_type] = 0
            tree_type_counter[tree_type] += 1

        print(f'\nAdded a cluster containing {tree_amount} trees.')

    else:
        area = 0
        while area < cluster_area:
            pass

    return background, mask, free_area


# Not really Utils, only moved here so synth_forest.py looks cleaner
def contact_position(area, cluster_mask, tree):
    """Randomly selects a contact point on the cluster and the tree and
    calculates the center position required for the tree for these two points to match."""
    side = np.random.choice(np.where(area > 0)[0])  # 0-left, 1-right, 2-up, 3-down
    contact = False
    if side == 0 or side == 1:
        x_pos = -1 * side  # so 0 or -1
        y_pos = np.random.choice(cluster_mask.shape[1])
        half = y_pos > cluster_mask.shape[1] / 2  # False: left half, True: right half

        while not contact:
            if cluster_mask[x_pos, y_pos] != 0:
                contact = True
            else:
                if x_pos >= 0:
                    x_pos += 1
                else:
                    x_pos -= 1
    else:
        x_pos = np.random.choice(cluster_mask.shape[0])
        y_pos = -1 * (side - 2)
        half = x_pos > cluster_mask.shape[0] / 2  # False: upper half, True: lower half

        while not contact:
            if cluster_mask[x_pos, y_pos] != 0:
                contact = True
            else:
                if y_pos >= 0:
                    y_pos += 1
                else:
                    y_pos -= 1

    if x_pos < 0:
        x_pos = cluster_mask.shape[0] + x_pos

    if y_pos < 0:
        y_pos = cluster_mask.shape[1] + y_pos

    cluster_contact = [x_pos, y_pos]

    if side == 0 or 2:
        tree_side = side + 1
    else:
        tree_side = side - 1

    contact = False
    if tree_side == 0 or tree_side == 1:
        x_pos = -1 * tree_side
        y_pos = np.random.choice(tree.shape[1] // 2)
        y_pos = y_pos + y_pos * half

        while not contact:
            if tree[x_pos, y_pos, 0] != 0:
                contact = True
            else:
                if x_pos >= 0:
                    x_pos += 1
                else:
                    x_pos -= 1
    else:
        x_pos = np.random.choice(tree.shape[0] // 2)
        y_pos = -1 * (tree_side - 2)
        x_pos = x_pos + x_pos * half

        while not contact:
            if tree[x_pos, y_pos, 0] != 0:
                contact = True
            else:
                if y_pos >= 0:
                    y_pos += 1
                else:
                    y_pos -= 1

    if x_pos < 0:
        x_pos = tree.shape[0] + x_pos

    if y_pos < 0:
        y_pos = tree.shape[1] + y_pos

    tree_contact = [x_pos, y_pos]

    x = area[0] + cluster_contact[0] + tree.shape[0] // 2 - tree_contact[0]
    y = area[2] + cluster_contact[1] + tree.shape[1] // 2 - tree_contact[1]

    return x, y

# notes:    - add blocked area for cluster (maybe draw circles around all pixels?)
#           - add cluster based on area function (count pixels)
#
# issues:   - clusters can run out of space with no safety measure
#           - cluster function moves trees from the side to the cluster like tetris --> creates holes
#           - clusters create no blocked area
#           - trees can still overlap with clustering, as contact is only checked on a single point
#           - trees are simply stacked on top of each other, not considering height
