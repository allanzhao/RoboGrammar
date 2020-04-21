'''
class DesignGraph

A design graph includes: adjacent matrix, node features, mask

Two preprocessing:
    1. The adjacent matrix is processed to be symmetric (undirected graph).
    2. The size of the graph has been padded to be max_nodes.

'''
import numpy as np
import quaternion

class DesignGraphState:
    def __init__(self, max_nodes, adj_matrix = None, features = None, masks = None, robot = None):
        self.max_nodes = max_nodes
        if adj_matrix is not None and features is not None and masks is not None:
            self.adj_matrix = deepcopy(adj_matrix)
            self.features = deepcopy(features)
            self.masks = deepcopy(masks)
        elif robot is not None:
            self.adj_matrix, self.features, self.masks = self.construct_from_robot(robot)
        else:
            print_error('DesignGraph cannot be constructed from None type.')
    
    '''
    construct the adjacent matrix and features from a robot instance.
    
    Note: copy from parse_log_file.py line 106 ~ line 145
    '''
    def construct_from_robot(self, robot):
        # Find the world position and rotation of links
        pos_rot = []
        for i, link in enumerate(robot.links):
            if link.parent >= 0:
                parent_pos, parent_rot = pos_rot[link.parent]
                parent_link_length = robot.links[link.parent].length
            else:
                parent_pos, parent_rot = np.zeros(3), np.quaternion(1, 0, 0, 0)
                parent_link_length = 0

            offset = np.array([parent_link_length * link.joint_pos, 0, 0])
            rel_pos = quaternion.rotate_vectors(parent_rot, offset)
            rel_rot = np_quaternion(link.joint_rot).conjugate()
            pos = parent_pos + rel_pos
            rot = parent_rot * rel_rot
            pos_rot.append((pos, rot))

        # Generate adjacency matrix
        adj_matrix = np.zeros((len(robot.links), len(robot.links)))
        for i, link in enumerate(robot.links):
            if link.parent >= 0:
                adj_matrix[link.parent, i] += 1

        # Generate features for links
        # Note: we can work with either the graph or the robot kinematic tree, but
        # the kinematic tree provides more information
        link_features = []
        for i, link in enumerate(robot.links):
            world_pos, world_rot = pos_rot[i]
            world_joint_axis = quaternion.rotate_vectors(world_rot, link.joint_axis)
            label_vec = np.zeros(len(all_labels))
            label_vec[all_labels.index(link.label)] = 1

            link_features.append(np.array([
                *featurize_link(link),
                *world_pos,
                *quaternion_coords(world_rot),
                *world_joint_axis,
                *label_vec]))
        link_features = np.array(link_features)
        
        real_size = link_features.shape[0]

        # make adj_matrix symmetric
        adj_matrix = adj_matrix + np.transpose(adj_matrix)

        # add blank nodes
        self.pad(adj_matrix, (self.max_nodes, self.max_nodes))
        self.pad(link_features, (self.max_nodes, link_features.shape[1]))

        # create mask
        masks = np.array([True if i < real_size else False for i in range(max_nodes)])

        return adj_matrix, link_features, masks

    def pad(self, array, shape):
        """
        array: Array to be padded
        reference: Reference array with the desired shape
        offsets: list of offsets (number of elements must be equal to the dimension of the array)
        """
        # Create an array of zeros with the reference shape
        result = np.zeros(shape)
        if len(shape) == 1:
            result[:array.shape[0], :] = array # ERROR: why result is 2d
        elif len(shape) == 2:
            result[:array.shape[0], :array.shape[1]] = array
        else:
            raise Exception('only 1 and 2d supported for now')
        return result