import argparse
from design_search import RobotDesignEnv, make_graph, build_normalized_robot, presimulate, simulate
import mcts
from neurons import NeuronStreamWrapper
import numpy as np
import os
import pyrobotdesign as rd
import random
import tasks
import time


class CameraTracker(object):
    def __init__(self, viewer, sim, robot_idx):
        self.viewer = viewer
        self.sim = sim
        self.robot_idx = robot_idx

        self.reset()

    def update(self, time_step):
        lower = np.zeros(3)
        upper = np.zeros(3)
        self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

        # Update camera position to track the robot smoothly
        target_pos = 0.5 * (lower + upper)
        camera_pos = self.viewer.camera_params.position.copy()
        camera_pos += 5.0 * time_step * (target_pos - camera_pos)
        self.viewer.camera_params.position = camera_pos

    def reset(self):
        lower = np.zeros(3)
        upper = np.zeros(3)
        self.sim.get_robot_world_aabb(self.robot_idx, lower, upper)

        self.viewer.camera_params.position = 0.5 * (lower + upper)


def set_joint_torques(sim, torques):
        torques /= np.linalg.norm(torques, ord=1)
        torques /= torques.max()
        # neural_input = 0.05 + 0.95 * (neural_input - neural_input.min()) / (neural_input.max() - neural_input.min())
        for link, _torques in zip(sim.get_robot(0).links, torques):
            link.joint_torque = _torques


def run_trajectory(sim, robot_idx, input_sequence, task, step_callback, neuron_stream_wrapper=None):
    step_callback(0)

    for j in range(input_sequence.shape[1]):
        if neuron_stream_wrapper is not None:
            neural_input = (np.expand_dims(neuron_stream_wrapper.neuron_stream.get_spike_frequencies(), axis=0) @ neuron_stream_wrapper.weights)[0]
            set_joint_torques(sim, neural_input)
        
        for k in range(task.interval):
            step_idx = j * task.interval + k
            sim.set_joint_targets(robot_idx, input_sequence[:,j].reshape(-1, 1))
            task.add_noise(sim, step_idx)
            sim.step()
            step_callback(step_idx + 1)


def view_trajectory(sim, robot_idx, input_sequence, task, no_view=False, neuron_stream_wrapper=None):
    record_step_indices = set()

    sim.save_state()

    viewer = rd.GLFWViewer()

    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    sim.get_robot_world_aabb(robot_idx, lower, upper)

    # Set initial camera parameters
    task_name = type(task).__name__
    if 'Ridged' in task_name or 'Gap' in task_name:
        viewer.camera_params.yaw = 0.0
    elif 'Wall' in task_name:
        viewer.camera_params.yaw = -np.pi / 2
    else:
        viewer.camera_params.yaw = -np.pi / 4
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 1.5 * np.linalg.norm(upper - lower)
        
    tracker = CameraTracker(viewer, sim, robot_idx)

    j = 0
    k = 0
    
    if neuron_stream_wrapper is not None:
        neural_input = (np.expand_dims(neuron_stream_wrapper.neuron_stream.get_spike_frequencies(), axis=0) @ neuron_stream_wrapper.weights)[0]
        set_joint_torques(sim, neural_input)
    
    sim_time = time.time()
    while not viewer.should_close():
        current_time = time.time()
        while sim_time < current_time:
            step_idx = j * task.interval + k
            
            if input_sequence is not None:
                sim.set_joint_targets(robot_idx, input_sequence[:,j].reshape(-1, 1))
            
            task.add_noise(sim, step_idx)
            sim.step()
            tracker.update(task.time_step)
            viewer.update(task.time_step)
            
            if viewer.camera_controller.should_record():
                record_step_indices.add(step_idx)
            
            sim_time += task.time_step
            k += 1
            
            if k >= task.interval:
                if neuron_stream_wrapper is not None:
                    neural_input = (np.expand_dims(neuron_stream_wrapper.neuron_stream.get_spike_frequencies(), axis=0) @ neuron_stream_wrapper.weights)[0]
                    set_joint_torques(sim, neural_input)
                j += 1
                k = 0
                    
            if input_sequence is not None and j >= input_sequence.shape[1]:
                j = 0
                k = 0
                sim.restore_state()
                tracker.reset()
        viewer.render(sim)
        
        if no_view:
            break

    sim.restore_state()
    
    if neuron_stream_wrapper:
        neuron_stream_wrapper.stop()

    return viewer.camera_params, record_step_indices

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

def main():
    parser = argparse.ArgumentParser(description="Robot design viewer.")
    parser.add_argument("task", type=str, help="Task (Python class name)")
    parser.add_argument("grammar_file", type=str, help="Grammar file (.dot)")
    parser.add_argument("rule_sequence", nargs="+", help="Rule sequence to apply")
    parser.add_argument("-o", "--optim", default=False, action="store_true", help="Optimize a trajectory")
    parser.add_argument("-s", "--opt_seed", type=int, default=None, help="Trajectory optimization seed")
    parser.add_argument("-e", "--episodes", type=int, default=1, help="Number of optimization episodes")
    parser.add_argument("-j", "--jobs", type=int, required=True, help="Number of jobs/threads")
    parser.add_argument("--input_sequence_file", type=str, help="File to save input sequence to (.csv)")
    parser.add_argument("--save_obj_dir", type=str, help="Directory to save .obj files to")
    parser.add_argument("--save_video_file", type=str, help="File to save video to (.mp4)")
    parser.add_argument("-l", "--episode_len", type=int, default=128, help="Length of episode")
    parser.add_argument("--no_view", action="store_true", help="Whether to open a window with simulation rendering.")
    args = parser.parse_args()

    task_class = getattr(tasks, args.task)
    task = task_class(episode_len=args.episode_len)
    graphs = rd.load_graphs(args.grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]

    rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
    if args.opt_seed is not None:
        opt_seed = args.opt_seed
    else:
        opt_seed = random.getrandbits(32)
        print("Using optimization seed:", opt_seed)

    graph = make_graph(rules, rule_sequence)
    robot = build_normalized_robot(graph)
    finalize_robot(robot)
    
    args.task_creating_kwargs = {"episode_len": args.episode_len}
    
    # neuron_stream_wrapper = NeuronStreamWrapper(stream_kwargs=dict(channels=32, raw_values_buffer_ms=task.horizon * task.interval * task.time_step * 1000))
    # neuron_stream_wrapper.start()
    neuron_stream_wrapper = None
    if args.optim:
        input_sequence, result = simulate(robot, task, opt_seed, args, neuron_stream_wrapper=neuron_stream_wrapper)
        print("Result:", result)
    else:
        input_sequence = None

    if args.input_sequence_file and input_sequence is not None:
        import csv
        with open(args.input_sequence_file, 'w', newline='') as input_seq_file:
            writer = csv.writer(input_seq_file)
            for col in input_sequence.T:
                writer.writerow(col)
        print("Saved input sequence to file:", args.input_sequence_file)

    robot_init_pos, has_self_collision = presimulate(robot)

    if has_self_collision:
        print("Warning: robot self-collides in initial configuration")

    main_sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(main_sim)
    # Rotate 180 degrees around the y axis, so the base points to the right
    main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    robot_idx = main_sim.find_robot_index(robot)

    camera_params, record_step_indices = view_trajectory(main_sim, robot_idx, input_sequence, task, no_view=args.no_view, neuron_stream_wrapper=neuron_stream_wrapper)

    if args.save_obj_dir is not None and input_sequence is not None:
        import export_mesh

        if record_step_indices:
            print("Saving .obj files for {} steps".format(len(record_step_indices)))

        os.makedirs(args.save_obj_dir, exist_ok=True)

        # Save the props/terrain once
        obj_file_name = os.path.join(args.save_obj_dir, 'terrain.obj')
        mtl_file_name = os.path.join(args.save_obj_dir, 'terrain.mtl')
        with open(obj_file_name, 'w') as obj_file, \
                 open(mtl_file_name, 'w') as mtl_file:
            dumper = export_mesh.ObjDumper(obj_file, mtl_file)
            obj_file.write("mtllib {}\n".format(os.path.split(mtl_file_name)[-1]))
            for prop_idx in range(main_sim.get_prop_count()):
                export_mesh.dump_prop(prop_idx, main_sim, dumper)
            dumper.finish()

        # Save the robot once per step
        def save_obj_callback(step_idx):
            if record_step_indices:
                if step_idx not in record_step_indices:
                    return
            else:
                if step_idx % 128 != 0:
                    return

            obj_file_name = os.path.join(args.save_obj_dir,
                                                                     'robot_{:04}.obj'.format(step_idx))
            # Use one .mtl file for all steps
            mtl_file_name = os.path.join(args.save_obj_dir, 'robot.mtl')
            with open(obj_file_name, 'w') as obj_file, \
                     open(mtl_file_name, 'w') as mtl_file:
                dumper = export_mesh.ObjDumper(obj_file, mtl_file)
                obj_file.write("mtllib {}\n".format(os.path.split(mtl_file_name)[-1]))
                export_mesh.dump_robot(robot_idx, main_sim, dumper)
                dumper.finish()

        run_trajectory(main_sim, robot_idx, input_sequence, task, save_obj_callback)

    if args.save_video_file and input_sequence is not None:
        import cv2

        if record_step_indices:
            print("Saving video for {} steps".format(len(record_step_indices)))

        viewer = rd.GLFWViewer()

        # Copy camera parameters from the interactive viewer
        viewer.camera_params = camera_params

        tracker = CameraTracker(viewer, main_sim, robot_idx)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video_file, fourcc, 60.0,
                                                         viewer.get_framebuffer_size())
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

        def write_frame_callback(step_idx):
                tracker.update(task.time_step)

                # 240 steps/second / 4 = 60 fps
                if step_idx % 4 == 0:
                        # Flip vertically, convert RGBA to BGR
                        frame = viewer.render_array(main_sim)[::-1,:,2::-1]
                        writer.write(frame)

        run_trajectory(main_sim, robot_idx, input_sequence, task, write_frame_callback, neuron_stream_wrapper=neuron_stream_wrapper)

        writer.release()

if __name__ == '__main__':
    main()
