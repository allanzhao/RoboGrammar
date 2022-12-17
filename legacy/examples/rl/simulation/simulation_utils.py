import sys, os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'design_search'))

from copy import deepcopy
import numpy as np

from design_search import presimulate, simulate, build_normalized_robot, make_initial_graph
import pyrobotdesign as rd

def make_sim_fn(task, robot, robot_init_pos):
    sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    return sim

def finalize_robot(robot):
    for link in robot.links:
        link.label = ""
        link.joint_label = ""
        if link.shape == rd.LinkShape.NONE:
            link.shape = rd.LinkShape.CAPSULE
            link.length = 0.1
            link.radius = 0.025
            link.color = [1.0, 0.0, 1.0]
        if link.joint_type == rd.JointType.NONE:
            link.joint_type = rd.JointType.FIXED
            link.joint_color = [1.0, 0.0, 1.0]
        
def build_robot(args):
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]

    graph = make_initial_graph()
    for r in rule_sequence:
        matches = rd.find_matches(rules[r].lhs, graph)
        if matches:
            graph = rd.apply_rule(rules[r], graph, matches[0])

    robot = build_normalized_robot(graph)
    finalize_robot(robot)

    return robot

def get_robot_state(sim, robot_id):
    base_tf = np.zeros((4, 4), order = 'f')
    sim.get_link_transform(robot_id, 0, base_tf)
    base_R = deepcopy(base_tf[0:3, 0:3])
    base_pos = deepcopy(base_tf[0:3, 3])

    # anguler velocity first and linear velocity next
    base_vel = np.zeros(6, order = 'f')
    sim.get_link_velocity(robot_id, 0, base_vel)
    
    n_dofs = sim.get_robot_dof_count(robot_id)
    
    joint_pos = np.zeros(n_dofs, order = 'f')
    sim.get_joint_positions(robot_id, joint_pos)
    
    joint_vel = np.zeros(n_dofs, order = 'f')
    sim.get_joint_velocities(robot_id, joint_vel)
    
    state = np.hstack((base_R.flatten(), base_pos, base_vel, joint_pos, joint_vel))

    return state

def presimulate(robot):
    """Find an initial position that will place the robot on the ground behind the
    x=0 plane, and check if the robot collides in its initial configuration."""
    temp_sim = rd.BulletSimulation()
    temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    temp_sim.step()
    robot_idx = temp_sim.find_robot_index(robot)
    lower = np.zeros(3)
    upper = np.zeros(3)
    temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
    return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)