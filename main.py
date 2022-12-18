import traceback
from time import time, sleep

import pyrobotdesign as rd

from env import SimEnvWrapper
from mppi import MPPI
from neurons import NeuronStream
from tasks import FlatTerrainTask
from utils import (build_normalized_robot, finalize_robot,
                   get_make_sim_and_task_fn, make_graph, presimulate)
from constants import *
from view import prepare_viewer, viewer_step


if __name__ == "__main__":
    
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
    make_sim_and_task_fn = get_make_sim_and_task_fn(task, robot, robot_init_pos=robot_init_pos)
    main_env, _ = make_sim_and_task_fn()
    env = SimEnvWrapper(make_sim_and_task_fn)
    
    dof_count = env.env.get_robot_dof_count(0)
    objective_fn = task.get_objective_fn()
    n_samples = 512 // NUM_THREADS
    
    # initialize controller
    controller = None
    # initialize neuron stream
    neuron_stream = None # NeuronStream(channels=CHANNELS, dt=DT)
    # initialize rendering
    viewer, tracker = prepare_viewer(main_env)
    
    if OPTIMIZE:
        optimizer = MPPI(env, HORIZON, n_samples, 
                            num_cpu=NUM_THREADS,
                            kappa=1.0,
                            gamma=task.discount_factor,
                            default_act="mean",
                            filter_coefs=ACTION_FILTER_COEFS,
                            seed=SEED,
                            neuron_stream=neuron_stream)
        
        # search for initial paths
        paths = optimizer.do_rollouts(SEED)
        optimizer.update(paths)
        optimizer.paths_per_cpu = 64 // NUM_THREADS
    else:
        optimizer = None
    
    try:
        prev_time = time()
        while True:
            
            if optimizer is not None:
                paths = optimizer.do_rollouts(SEED + len(optimizer.sol_act) + 1)
                optimizer.update(paths)
                
                actions = optimizer.act_sequence[0]
                print(actions.shape)
                
                optimizer.advance_time()
                step = len(optimizer.sol_act)
            else:
                actions = np.zeros(dof_count)
                step = 0
            
                        
            if viewer is not None:
                viewer_step(main_env, task, actions, viewer, tracker, torques=np.zeros_like(actions))
            
            if controller is not None:
                pass
            
            curr_time = time()
            print("step =", step, "\ttime =", curr_time - prev_time, "\tactions =", actions)
            
            sleep_time = curr_time - prev_time
            sleep((1/ 15 - sleep_time + 0.01) if sleep_time < 1 / 15 else 0.01)
            prev_time = curr_time
    
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