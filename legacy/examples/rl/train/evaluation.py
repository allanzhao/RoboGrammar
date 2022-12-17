import numpy as np
import torch
import time

import gym
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from common.common import *

import pyrobotdesign as rd

def evaluate(args, actor_critic, ob_rms, env_name, seed, num_processes, device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, None, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < args.eval_num:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float64,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

def render(render_env, actor_critic, ob_rms, deterministic = False, repeat = False):
    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)

    viewer = rd.GLFWViewer()
    viewer.camera_params.position = 0.5 * (lower + upper)
    viewer.camera_params.yaw = 0.0
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)

    time_step = render_env.task.time_step * render_env.frame_skip

    while True:
        total_reward = 0.
        sim_time = 0.
        render_time_start = time.time()
        with torch.no_grad():
            ob = render_env.reset()
            done = False
            episode_length = 0
            while not done:
                ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                _, u, _, _ = actor_critic.act(torch.tensor(ob).unsqueeze(0), None, None, deterministic = deterministic)
                u = u.detach().squeeze(dim = 0).numpy()
                ob, reward, done, _ = render_env.step(u)
                total_reward += reward

                episode_length += 1
                
                # render
                render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)
                target_pos = 0.5 * (lower + upper)
                camera_pos = viewer.camera_params.position.copy()
                camera_pos += 5.0 * time_step * (target_pos - camera_pos)
                viewer.camera_params.position = camera_pos
                viewer.update(time_step)
                viewer.render(render_env.sim)
                sim_time += time_step
                render_time_now = time.time()
                if render_time_now - render_time_start < sim_time:
                    time.sleep(sim_time - (render_time_now - render_time_start))
        
        print_info('rendering:')

        print_info('length = ', episode_length)
        print_info('total reward = ', total_reward)
        print_info('avg reward = ', total_reward / (episode_length * render_env.frame_skip))
        
        if not repeat:
            break
    
    del viewer
    
# render each sub-step
def render_full(render_env, actor_critic, ob_rms, deterministic = False, repeat = False):
    # Get robot bounds
    lower = np.zeros(3)
    upper = np.zeros(3)
    render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)

    viewer = rd.GLFWViewer()
    viewer.camera_params.position = 0.5 * (lower + upper)
    viewer.camera_params.yaw = 0.0
    viewer.camera_params.pitch = -np.pi / 6
    viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)

    time_step = render_env.task.time_step

    control_frequency = render_env.frame_skip
    render_env.set_frame_skip(1)
    
    while True:
        total_reward = 0.
        sim_time = 0.
        render_time_start = time.time()
        with torch.no_grad():
            ob = render_env.reset()
            done = False
            episode_length = 0
            while episode_length < 128 * control_frequency:
                if episode_length % control_frequency == 0:
                    ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                    _, u, _, _ = actor_critic.act(torch.tensor(ob).unsqueeze(0), None, None, deterministic = deterministic)
                    u = u.detach().squeeze(dim = 0).numpy()
        
                ob, reward, done, _ = render_env.step(u)
                total_reward += reward

                episode_length += 1

                # render
                render_env.sim.get_robot_world_aabb(render_env.robot_index, lower, upper)
                target_pos = 0.5 * (lower + upper)
                camera_pos = viewer.camera_params.position.copy()
                camera_pos += 20.0 * time_step * (target_pos - camera_pos)
                sim_time += time_step
                render_time_now = time.time()
                
                if render_time_now - render_time_start < sim_time:
                    time.sleep(sim_time - (render_time_now - render_time_start))
            
                if sim_time + time_step > render_time_now - render_time_start:
                    viewer.camera_params.position = camera_pos
                    viewer.update(time_step)
                    viewer.render(render_env.sim)
        
        print_info('rendering:')

        print_info('length = ', episode_length)
        print_info('total reward = ', total_reward)
        print_info('avg reward = ', total_reward / (episode_length * render_env.frame_skip))
        
        if not repeat:
            break
    
    del viewer