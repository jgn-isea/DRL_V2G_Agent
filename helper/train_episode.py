# -*- coding = utf-8 -*-
# @Time: 04.01.2025 23:32
# @Author: J.Gong
# @File: train_episode
# @Software: PyCharm

from helper.decorator import live_plot
import time
import random
import tensorflow as tf


@live_plot
def train_episode(agent, env, episode, reward_log, time_log, soc_log, epi_random, reference_values, plot_elements, **kwargs):
    start_time = time.time()
    state = env.reset(episode)
    total_reward = 0
    is_ppo_agent = "PPO" in type(agent).__name__
    action_size = env.action_size
    max_timesteps = env.max_timesteps

    for t in range(max_timesteps):
        if episode < epi_random:
            if action_size == 1:
                action = [random.uniform(-1, 1)]  # continuous action space
                if is_ppo_agent:
                    log_prob = 0  # dummy value for log_prob
            else:
                action = random.randint(0, action_size - 1)  # discrete action space
        else:
            if is_ppo_agent:  # PPO agent
                action, log_prob = agent.choose_action(state)
            else:  # other agents
                action = agent.choose_action(state)

        next_state, reward, done, soc, _ = env.step(action)

        if is_ppo_agent:  # PPO agent
            agent.trajectory_buffer.store(state, action, reward, next_state, done, log_prob)
        else:  # other agents
            agent.replay_buffer.store(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        if done:
            break

    reward_log.append(total_reward)
    time_log.append(time.time() - start_time)
    soc_log.append(soc)

    # Train the agent
    agent.learn()
    agent.save_var_hyperparameters()

    return total_reward


@live_plot
def train_episode_mask(agent, env, episode, reward_log, time_log, soc_log, epi_random, reference_values, plot_elements,
                       use_prediction, **kwargs):
    start_time = time.time()
    state = env.reset(episode, use_prediction=use_prediction)
    mask = env.get_mask()
    total_reward = 0
    is_ppo_agent = "PPO" in type(agent).__name__
    action_size = env.action_size
    max_timesteps = env.max_timesteps

    for t in range(max_timesteps):
        if episode < epi_random:
            if action_size == 1:
                action = [random.uniform(-1, 1)]  # continuous action space
                if is_ppo_agent:
                    log_prob = 0  # dummy value for log_prob
            else:
                action = random.randint(0, action_size - 1)  # discrete action space
        else:
            if is_ppo_agent:  # PPO agent
                action, log_prob = agent.choose_action(state, mask)
            else:  # other agents
                action = agent.choose_action(state, mask)

        next_state, reward, done, soc, _ = env.step(action, use_prediction=use_prediction)
        next_mask = env.get_mask()

        if is_ppo_agent:  # PPO agent
            agent.trajectory_buffer.store(state, action, reward, next_state, done, log_prob, mask, next_mask)
        else:  # other agents
            agent.replay_buffer.store(state, action, reward, next_state, done, mask, next_mask)

        state = next_state
        mask = next_mask
        total_reward += reward
        if done:
            break

    reward_log.append(total_reward)
    time_log.append(time.time() - start_time)
    soc_log.append(soc)

    # Train the agent
    agent.learn()
    agent.save_var_hyperparameters()

    return total_reward