# import python packages
import ast
import csv
import numpy as np

# import our own packages
from Preprocessor import Preprocessor
import pyrobotdesign as rd
from design_search import build_normalized_robot, make_initial_graph

'''
load_terminal_design_data

function to load all terminal designs and their rewards from csv file
'''
def load_terminal_design_data(raw_dataset_path, grammar_file):
    graphs = rd.load_graphs(grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.label)
    all_labels = sorted(list(all_labels))

    preprocessor = Preprocessor(all_labels = all_labels)

    with open(raw_dataset_path, newline='') as log_file:
        reader = csv.DictReader(log_file)
        
        all_link_features = []
        all_link_adj = []
        all_results = []
        max_nodes = 0
        for row in reader:
            rule_seq = ast.literal_eval(row['rule_seq'])
            result = float(row['result'])

            all_results.append(result)

            # Build a robot from the rule sequence
            robot_graph = make_initial_graph()
            for r in rule_seq:
                matches = rd.find_matches(rules[r].lhs, robot_graph)
                # Always use the first match
                robot_graph = rd.apply_rule(rules[r], robot_graph, matches[0])
            
            adj_matrix, link_features, _ = preprocessor.preprocess(robot_graph)

            all_link_features.append(link_features)
            all_link_adj.append(adj_matrix)

            max_nodes = max(max_nodes, adj_matrix.shape[0])

        all_adj_matrix_pad, all_link_features_pad, all_masks = [], [], []
        for adj_matrix, link_features in zip(all_link_adj, all_link_features):
            adj_matrix_pad, link_features_pad, masks = preprocessor.pad_graph(adj_matrix, link_features, max_nodes = max_nodes)
            all_adj_matrix_pad.append(adj_matrix_pad)
            all_link_features_pad.append(link_features_pad)
            all_masks.append(masks)
    
    return all_link_features_pad, all_adj_matrix_pad, all_masks, all_results

'''
update_memory

update the maximum reward under the subtree of robot_graph. Insert a new node into memory for the first time.
'''
def update_memory(memory, preprocessor, robot_graph, result):
    robot_hash_key = hash(robot_graph)

    if robot_hash_key not in memory.keys():
        adj_matrix, link_features, _ = preprocessor.preprocess(robot_graph)
        memory[robot_hash_key] = {'adj_matrix': adj_matrix, 'link_features': link_features, 'V': -np.inf, 'hit': 0}
    
    memory[robot_hash_key]['V'] = max(memory[robot_hash_key]['V'], result)
    memory[robot_hash_key]['hit'] += 1

'''
load_partial_design_data

function to load all designs including partial designs and terminal designs, and their rewards, from csv file
'''
def load_partial_design_data(raw_dataset_path, grammar_file):
    graphs = rd.load_graphs(grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.label)
    all_labels = sorted(list(all_labels))

    preprocessor = Preprocessor(all_labels = all_labels)

    with open(raw_dataset_path, newline='') as log_file:
        reader = csv.DictReader(log_file)

        memory = dict() 
        idx = 0
        for row in reader:
            if idx % 1000 == 0:
                print(f'processing idx = {idx}')
            idx += 1

            rule_seq = ast.literal_eval(row['rule_seq'])
            result = float(row['result'])

            # Build a robot from the rule sequence
            robot_graph = make_initial_graph()
            update_memory(memory, preprocessor, robot_graph, result)
            for r in rule_seq:
                matches = rd.find_matches(rules[r].lhs, robot_graph)
                # Always use the first match
                robot_graph = rd.apply_rule(rules[r], robot_graph, matches[0])
                update_memory(memory, preprocessor, robot_graph, result)
                
        initial_robot_graph = make_initial_graph()
        print('#hit on initial state: ', memory[hash(initial_robot_graph)]['hit'])

        all_link_features = []
        all_link_adj = []
        all_results = []
        max_nodes = 0
        for _, robot_hash_key in enumerate(memory):
            adj_matrix, link_features, result = \
                memory[robot_hash_key]['adj_matrix'], memory[robot_hash_key]['link_features'], memory[robot_hash_key]['V']

            all_link_features.append(link_features)
            all_link_adj.append(adj_matrix)
            all_results.append(result)

            max_nodes = max(max_nodes, adj_matrix.shape[0])

        all_adj_matrix_pad, all_link_features_pad, all_masks = [], [], []
        for adj_matrix, link_features in zip(all_link_adj, all_link_features):
            adj_matrix_pad, link_features_pad, masks = preprocessor.pad_graph(adj_matrix, link_features, max_nodes = max_nodes)
            all_adj_matrix_pad.append(adj_matrix_pad)
            all_link_features_pad.append(link_features_pad)
            all_masks.append(masks)
    
    return all_link_features_pad, all_adj_matrix_pad, all_masks, all_results