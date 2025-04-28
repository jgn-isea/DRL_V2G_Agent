# -*- coding = utf-8 -*-
# @Time: 04.01.2025 23:31
# @Author: J.Gong
# @File: evaluate_policy
# @Software: PyCharm
import numpy as np
from helper.decorator import live_plot


@live_plot
def evaluate_policy(env, agent, eval_log, reference_values=None, plot_elements=None, **kwargs):
    """
    Evaluate the current policy over test_data and return the total and average reward.
    """
    rewards_epi = []
    soc_at_departure = []
    episodes = len(env.data_test)
    for e in range(episodes):
        rewards = 0
        state = env.reset(e, eval=True)
        ref_aging_reward = env.AgingModel.reward_memory
        done = False
        while not done:
            action = agent.choose_action(state, deterministic=True)
            state, reward, done, soc, power = env.step(action)
            rewards += reward
        soc_at_departure.append(soc)
        rewards_epi.append(rewards + ref_aging_reward)
    eval_log.append((rewards_epi, soc_at_departure))
    return np.sum(rewards_epi), np.mean(rewards_epi)


@live_plot
def evaluate_policy_mask(env, agent, eval_log, reference_values, plot_elements, use_prediction, **kwargs):
    """
    Evaluate the current policy over test_data and return the total and average reward.
    """
    rewards_epi = []
    soc_at_departure = []
    episodes = len(env.data_test)
    for e in range(episodes):
        rewards = 0
        state = env.reset(e, eval=True, use_prediction=use_prediction)
        mask = env.get_mask()
        ref_aging_reward = env.AgingModel.reward_memory
        done = False
        while not done:
            action = agent.choose_action(state, mask, deterministic=True)
            state, reward, done, soc, power = env.step(action, use_prediction=use_prediction)
            mask = env.get_mask()
            rewards += reward
        soc_at_departure.append(soc)
        rewards_epi.append(rewards + ref_aging_reward)
    eval_log.append((rewards_epi, soc_at_departure))
    return np.sum(rewards_epi), np.mean(rewards_epi)
