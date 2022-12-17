import multiprocessing as mp
import traceback
from time import sleep

import numpy as np
import pyrobotdesign as rd

from env import SimEnvWrapper
from mppi import MPPI
from neurons import NeuronStream
from tasks import FlatTerrainTask
from utils import (build_normalized_robot, finalize_robot,
                   get_make_sim_and_task_fn, make_graph, presimulate)


if __name__ == "__main__":
    GRAMMAR_FILEPATH = "./data/designs/grammar_apr30_asym.dot"
    RULE_SEQUENCE = "0, 7, 1, 12, 1, 13, 10, 6, 2, 4, 16, 3, 16, 4, 19, 17, 9, 2, 4, 5, 5, 4, 5, 16, 16, 4, 17, 18, 9, 14, 5, 8, 9, 8, 9, 9, 8, 8".split(" ")
    CHANNELS = 16
    DT = 16 / 240
    HORIZON = 40
    NUM_THREADS = mp.cpu_count() - 1
    SEED = np.random.randint(10000000)
    
    # initialize controller
    controller = None
    # initialize neuron stream
    neuron_stream = NeuronStream(channels=CHANNELS, dt=DT)
    # initialize input sequence
    input_sequence = []
    # initialize rendering function
    render = None
    
    # initialize task
    task = FlatTerrainTask()
    graphs = rd.load_graphs(GRAMMAR_FILEPATH)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s.strip(",")) for s in RULE_SEQUENCE]
    
    # initialize graph
    graph = make_graph(rules, rule_sequence)
    # initialize robot
    robot = build_normalized_robot(graph)
    finalize_robot(robot)
    robot_init_pos, _ = presimulate(robot)
    
    # initialize env
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