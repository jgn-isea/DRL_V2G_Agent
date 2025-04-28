# -*- coding = utf-8 -*-
# @Time: 11.01.2025 20:17
# @Author: J.Gong
# @File: load_agent
# @Software: PyCharm
import os
import warnings

import pandas as pd


def load_model(agent, env, model_path):
    agent_type = type(agent).__name__
    if "SAC" in agent_type:
        model_path = os.path.join("SAC_mdl", model_path) if not model_path.startswith("SAC_mdl/") else model_path
    elif "DDPG" in agent_type:
        model_path = os.path.join("DDPG_mdl", model_path) if not model_path.startswith("DDPG_mdl/") else model_path
    elif "DQN" in agent_type:
        model_path = os.path.join("DQN_mdl", model_path) if not model_path.startswith("DQN_mdl/") else model_path
    elif "PPO" in agent_type:
        model_path = os.path.join("PPO_mdl", model_path) if not model_path.startswith("PPO_mdl/") else model_path
    elif "TD3" in agent_type:
        model_path = os.path.join("TD3_mdl", model_path) if not model_path.startswith("TD3_mdl/") else model_path
    else:
        raise NameError(f"Agent type {agent_type} unrecognized. ")
    # Load a pre-trained DQN model network, the replaybuffer and the hyperparameters
    hyperparameter = pd.read_pickle(os.path.join(model_path, "final_hyperparameters.pkl"))
    hidden_width = hyperparameter["hidden_width"][0]
    hidden_depth = hyperparameter["hidden_depth"][0]
    if hidden_width != agent.hidden_width or hidden_depth != agent.hidden_depth:
        warnings.warn("The hidden width or depth of the model is different from the current model. Create new agent...")
        agent = agent.__class__(state_dim=env.state_size, action_dim=env.action_size, mask_dim=env.state_size,
                                hidden_width=hidden_width, hidden_depth=hidden_depth)
    agent.load_hyperparameters(model_path)
    agent.load_networks_replaybuffer(model_path, eval=False)

    try:
        agent.var_hyperparameters = pd.read_pickle(os.path.join(model_path, "var_hyperparameters.pkl"))
        agent.var_hyperparameters = agent.var_hyperparameters.to_dict(orient="list")
    except FileNotFoundError:
        warnings.warn("Variable hyperparameters not found. New variable hyperparameters will be generated.")
        agent.var_hyperparameters = {}

    # Load train and test data
    test_data_filepath = os.path.join(model_path, "dataset_test.pkl")
    train_data_filepath = os.path.join(model_path, "dataset_train.pkl")
    env.load_dataset(data_train=pd.read_pickle(train_data_filepath), data_test=pd.read_pickle(test_data_filepath))

    # Load logs
    training_data = pd.read_csv(os.path.join(model_path, "training_data.csv"))
    eval_data = pd.read_pickle(os.path.join(model_path, "evaluation_results.pkl"))

    reward_log = training_data["Total Reward"].tolist()
    time_log = training_data["Training Time (s)"].tolist()
    soc_log = training_data["SOC at Departure"].tolist()
    eval_log = eval_data[["Episode Rewards", "Episode final SOCs"]].values.tolist()

    try:
        reference_results = pd.read_pickle(os.path.join(model_path, "reference_results.pkl"))
    except FileNotFoundError:
        warnings.warn("Reference results not found. New reference results will be generated.")
        reference_results = None

    return reward_log, time_log, soc_log, eval_log, reference_results, agent
