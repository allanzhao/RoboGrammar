'''
class Preprocessor

convert a robot_graph into the data which can be input into GNN

Two preprocessing:
    1. build adjacent matrix and features
    2. pad to be max_nodes
'''
import numpy as np
import quaternion
from design_search import make_initial_graph, build_normalized_robot, get_applicable_matches, has_nonterminals

def np_quaternion(q):
    """Create a np.quaternion from a rd.Quaternion."""
    return np.quaternion(q.w, q.x, q.y, q.z)

def one_hot_encode(enum_member):
    """Encode an enum member as a one-hot vector."""
    vec = np.zeros(len(type(enum_member).__members__))
    vec[int(enum_member)] = 1
    return vec

def quaternion_coords(q):
    """Get the coefficients of a rd.Quaternion as an np.ndarray."""
    return np.array([q.w, q.x, q.y, q.z])

def featurize_link(link):
    """Extract a feature vector from a rd.Link."""
    return np.array([*one_hot_encode(link.joint_type),
                    link.joint_pos,
                    *quaternion_coords(link.joint_rot),
                    *link.joint_axis,
                    *one_hot_encode(link.shape),
                    link.length,
                    link.radius,
                    link.density,
                    link.friction,
                    link.joint_kp,
                    link.joint_kd,
                    link.joint_torque,
                    *one_hot_encode(link.joint_control_mode)])

class Preprocessor:
    def __init__(self, all_labels = None, max_nodes = None):
        self.max_nodes = max_nodes
        self.all_labels = all_labels
    
    '''
    preprocess the state data

    input a robot graph, output the adjacent matrix, features, and masks
    '''
    def preprocess(self, robot_graph, max_nodes = None):
        robot = build_normalized_robot(robot_graph)

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
            label_vec = np.zeros(len(self.all_labels))
            label_vec[self.all_labels.index(link.label)] = 1

            link_features.append(np.array([
                *featurize_link(link),
                *world_pos,
                *quaternion_coords(world_rot),
                *world_joint_axis,
                *label_vec]))
        link_features = np.array(link_features)

        # make adj_matrix symmetric
        adj_matrix = adj_matrix + np.transpose(adj_matrix)

        masks = None

        if max_nodes is None:
            max_nodes = self.max_nodes
        
        if max_nodes is not None:
            if max_nodes > len(link_features):
                adj_matrix, link_features, masks = self.pad_graph(adj_matrix, link_features, max_nodes)
            else:
                masks = np.full(len(link_features), True)
                
        return adj_matrix, link_features, masks

    def pad_graph(self, adj_matrix, features, max_nodes):
        real_size = features.shape[0]

        # add blank nodes
        adj_matrix = self.pad(adj_matrix, (max_nodes, max_nodes))
        features = self.pad(features, (max_nodes, features.shape[1]))

        # create mask
        masks = np.array([True if i < real_size else False for i in range(max_nodes)])

        return adj_matrix, features, masks

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
