import traceback
import os
from time import sleep
import numpy as np

import pyrobotdesign as rd

from tasks import FlatTerrainTask
from env import SimEnvWrapper
from neurons import NeuronStream
from mppi import MPPI


def get_make_sim_and_task_fn(robot, robot_init_pos):
    
    def fn():
        sim = rd.BulletSimulation(task.time_step)
        task.add_terrain(sim)
        # Rotate 180 degrees around the y axis, so the base points to the right
        sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        return sim, task
    
    return fn


def get_applicable_matches(rule, graph):
    """Generates all applicable matches for rule in graph."""
    for match in rd.find_matches(rule.lhs, graph):
        if rd.check_rule_applicability(rule, graph, match):
            yield match


def make_initial_graph():
    """Make an initial robot graph."""
    n0 = rd.Node()
    n0.name = 'robot'
    n0.attrs.label = 'robot'
    initial_graph = rd.Graph()
    initial_graph.nodes = [n0]
    return initial_graph


def make_graph(rules, rule_sequence):
    graph = make_initial_graph()
    for r in rule_sequence:
        matches = list(get_applicable_matches(rules[r], graph))
        if matches:
            graph = rd.apply_rule(rules[r], graph, matches[0])
        else:
            raise ValueError("Rule in sequence has no applicable matches")
    return graph


def build_normalized_robot(graph):
    """Build a robot from the graph and normalize the mass of the body links."""
    robot = rd.build_robot(graph)

    body_links = []
    total_body_length = 0.0
    for link in robot.links:
        if np.isclose(link.radius, 0.045):
            # Link is a body link
            body_links.append(link)
            total_body_length += link.length
            target_mass = link.length * link.density

    if body_links:
        body_density = target_mass / total_body_length
        for link in body_links:
            link.density = body_density

    return robot


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


def simulate_one(env, input_sequence, neuron_stream):
    pass


def simulate():
    pass


if __name__ == "__main__":
    GRAMMAR_FILEPATH = "./data/designs/grammar_apr30_asym.dot"
    RULE_SEQUENCE = "0, 7, 1, 12, 1, 13, 10, 6, 2, 4, 16, 3, 16, 4, 19, 17, 9, 2, 4, 5, 5, 4, 5, 16, 16, 4, 17, 18, 9, 14, 5, 8, 9, 8, 9, 9, 8, 8".split(" ")
    CHANNELS = 16
    DT = 16 / 240
    HORIZON = 40
    NUM_THREADS = os.cpu_count() - 1
    SEED = np.random.randint(10000000)
    
    # initialize controller
    controller = None
    # initialize neuron stream
    neuron_stream = NeuronStream(channels=CHANNELS, dt=DT)
    # initialize input sequence
    input_sequence = []
    # initialize rendering function
    render = None
    
    task = FlatTerrainTask()
    graphs = rd.load_graphs(GRAMMAR_FILEPATH)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s.strip(",")) for s in RULE_SEQUENCE]
    
    graph = make_graph(rules, rule_sequence)
    robot = build_normalized_robot(graph)
    finalize_robot(robot)
    
    robot_init_pos, _ = presimulate(robot)
    
    make_sim_and_task_fn = get_make_sim_and_task_fn(robot, robot_init_pos=robot_init_pos)
    env = SimEnvWrapper(make_sim_and_task_fn)
    
    dof_count = env.env.get_robot_dof_count(0)
    objective_fn = task.get_objective_fn()
    
    n_samples = 512
    optimizer = MPPI(env, HORIZON, n_samples // NUM_THREADS, 
                        num_cpu=NUM_THREADS,
                        kappa=1.0,
                        gamma=task.discount_factor,
                        default_act="mean",
                        filter_coefs=[0.10422766377112629, 0.3239870556899027, 0.3658903830367387, 0.3239870556899027, 0.10422766377112629],
                        seed=SEED,
                        neuron_stream=neuron_stream)
    
    try:
        while True:
            
            input_sequence = simulate_one(optimizer, objective_fn, env, input_sequence, neuron_stream)
            
            if render is not None:
                # render the simulation
                pass
            
            if controller is not None:
                # move the motors
                pass
            
            sleep(0.01)
    
    except KeyboardInterrupt:
        pass
    
    except:
        traceback.print_exc()
        
    finally:
        # stop the threaded processes.
        if controller is not None:
            controller.stop()
        if neuron_stream is not None:
            neuron_stream.stop()