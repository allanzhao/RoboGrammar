import argparse
import ast
import csv
from design_search import build_normalized_robot, make_initial_graph
import numpy as np
import pyrobotdesign as rd
import quaternion
import IPython

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

def featurize_node(node_attrs):
  """Extract a feature vector from a rd.NodeAttributes."""
  return np.array([*one_hot_encode(node_attrs.shape),
                   node_attrs.length,
                   node_attrs.radius,
                   node_attrs.density,
                   node_attrs.friction])

def featurize_edge(edge_attrs):
  """Extract a feature vector from a rd.EdgeAttributes."""
  return np.array([*one_hot_encode(edge_attrs.joint_type),
                   edge_attrs.joint_pos,
                   *quaternion_coords(edge_attrs.joint_rot),
                   *edge_attrs.joint_axis,
                   edge_attrs.joint_kp,
                   edge_attrs.joint_kd,
                   edge_attrs.joint_torque,
                   edge_attrs.joint_lower_limit,
                   edge_attrs.joint_upper_limit,
                   *one_hot_encode(edge_attrs.joint_control_mode),
                   edge_attrs.scale,
                   edge_attrs.mirror])

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

def main(log_file=None, grammar_file=None):
  parser = argparse.ArgumentParser(
      description="Example code for parsing a MCTS log file.")
  
  if not log_file or not grammar_file:
    parser.add_argument("log_file", type=str, help="Log file (.csv)")
    parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
    args = parser.parse_args()
  else:
    args = argparse.Namespace()
    args.grammar_file = grammar_file
    args.log_file = log_file

  graphs = rd.load_graphs(args.grammar_file)
  rules = [rd.create_rule_from_graph(g) for g in graphs]

  # Find all possible link labels, so they can be one-hot encoded
  all_labels = set()
  for rule in rules:
    for node in rule.lhs.nodes:
      all_labels.add(node.attrs.label)
  all_labels = sorted(list(all_labels))

  with open(args.log_file, newline='') as log_file:
    reader = csv.DictReader(log_file)
    
    all_link_features = []
    all_link_adj = []
    all_results = []
    for row in reader:
      full_rule_seq = ast.literal_eval(row['rule_seq'])
      result = float(row['result'])

      for prefix_len in range(len(full_rule_seq) + 1):
        rule_seq = full_rule_seq[:prefix_len]
        all_results.append(result)

        # Build a robot from the rule sequence
        robot_graph = make_initial_graph()
        for r in rule_seq:
          matches = rd.find_matches(rules[r].lhs, robot_graph)
          # Always use the first match
          robot_graph = rd.apply_rule(rules[r], robot_graph, matches[0])
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
          label_vec = np.zeros(len(all_labels))
          label_vec[all_labels.index(link.label)] = 1

          link_features.append(np.array([
              *featurize_link(link),
              *world_pos,
              *quaternion_coords(world_rot),
              *world_joint_axis,
              *label_vec]))
        link_features = np.array(link_features)

        all_link_features.append(link_features)
        all_link_adj.append(adj_matrix)

  return all_link_features, all_link_adj, all_results

if __name__ == '__main__':
  main()
