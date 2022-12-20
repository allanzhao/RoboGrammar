import traceback
from time import time, sleep
from datetime import datetime
import os


from env import SimEnvWrapper
from mppi import MPPI
from neurons import NeuronStream
from tasks import FlatTerrainTask
from utils import (build_normalized_robot, finalize_robot, convert_joint_angles,
                   get_make_sim_and_task_fn, make_graph, presimulate)
from constants import *
from view import prepare_viewer, viewer_step
from controller import Controller


if __name__ == "__main__":
    
    # initialize task
    # task = FlatTerrainTask()
    # graphs = rd.load_graphs(GRAMMAR_FILEPATH)
    # rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s.strip(",")) for s in RULE_SEQUENCE]
    
    # initialize graph
    # graph = make_graph(rules, rule_sequence)
    # initialize robot
    # robot = build_normalized_robot(graph)
    # finalize_robot(robot)
    # robot_init_pos, _ = presimulate(robot)
    
    # initialize env
    # make_sim_and_task_fn = get_make_sim_and_task_fn(task, robot, robot_init_pos=robot_init_pos)
    # main_env, _ = make_sim_and_task_fn()
    # env = SimEnvWrapper(make_sim_and_task_fn)
    
    # dof_count = main_env.get_robot_dof_count(0)
    dof_count = 11
    # objective_fn = task.get_objective_fn()
    n_samples = 512 // NUM_THREADS
    
    # initialize controller
    controller = Controller()
    # initialize neuron stream
    neuron_stream = None # NeuronStream(channels=CHANNELS, dt=DT)
    # initialize rendering
    viewer, tracker = None, None # prepare_viewer(main_env)
    
    # print(dir(robot))
    # print()
    # print(dir(main_env))
    # exit()
    
    # torques = np.ones(dof_count)
    # main_env.get_joint_motor_torques(0, torques)
    # print(torques)
    # main_env.add_joint_torques(0, np.ones(dof_count) * 10)
    # main_env.get_joint_motor_torques(0, torques)
    # print(torques)
    
    # set_joint_torques(main_env, np.ones(dof_count) * 0, norm=False)
    
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
    
    if INPUT_ACTION_SEQUENCE is not None:
        action_sequence = np.load(INPUT_ACTION_SEQUENCE)
        SAVE_ACTION_SEQUENCE = False
    
    if SAVE_ACTION_SEQUENCE:
        action_sequence = []
    
    try:
        prev_time = time()
        step = 0
        while True:
            
            if optimizer is not None:
                paths = optimizer.do_rollouts(SEED + len(optimizer.sol_act) + 1)
                optimizer.update(paths)
                
                actions = optimizer.act_sequence[0]
                
                optimizer.advance_time()
            elif INPUT_ACTION_SEQUENCE is not None:
                actions = action_sequence[step % len(action_sequence)]
            else:
                try:
                    with open("actions.csv", "r") as f:
                        actions = np.array(list(map(float, f.read().split(","))))
                    assert len(actions) == dof_count
                except:
                    actions = np.zeros(dof_count)
                
            # main_env.get_joint_motor_torques(0, torques)
            # print(get_joint_torques(main_env))    
            
            if SAVE_ACTION_SEQUENCE:
                action_sequence.append(actions)
                
            if viewer is not None:
                viewer_step(main_env, task, actions, viewer, tracker) # , torques=np.zeros_like(actions)) # np.random.rand(*actions.shape))
            
            if controller is not None:
                actions_t = convert_joint_angles(actions)
                controller.move(actions_t)
            
            curr_time = time()
            
            sleep_time = curr_time - prev_time
            print("step =", step, "\ttime =", np.round(sleep_time, 4), "\tactions =", np.round(actions, 2))
            print(actions_t)
            sleep((1 / 15 - sleep_time + 0.01) if sleep_time < 1 / 15 else 0.01)
            prev_time = curr_time
            step += 1
            
            # change this sleep HERE!
            sleep(1)
    
    except KeyboardInterrupt:
        pass
    
    except:
        traceback.print_exc()
        
    finally:
        # stop the threaded processes.
        if SAVE_ACTION_SEQUENCE:
            output_path = os.path.join("output", str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-") + ".npy")
            np.save(output_path, action_sequence)
        
        if controller is not None:
            controller.stop()
        if neuron_stream is not None:
            neuron_stream.stop()