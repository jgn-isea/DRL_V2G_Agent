# -*- coding = utf-8 -*-
# @Time: 04.01.2025 23:41
# @Author: J.Gong
# @File: save_results
# @Software: PyCharm

import os
import pandas as pd
from datetime import datetime


def save_results(
    reward_log,
    time_log,
    soc_log,
    eval_log,
    reference_results,
    env,
    agent,
    fig_train,
    fig_eval,
    args,
    rewards_filepath="training_data.csv",
    reference_results_filepath="reference_results.pkl",
    plot_training_filepath="training_plot.png",
    plot_eval_filepath="evaluation_plot.png",
    save_folder=None,
):
    """
    Save training results, logs, evaluation results, plots, and final models.

    Args:
        reward_log (list): Log of rewards for each episode.
        time_log (list): Log of training times for each episode.
        eval_log (list): Log of evaluation results (total reward, average reward).
        env (EVChargingEnvSAC): The environment used for training.
        agent (object): The trained SAC agent.
        fig(Figure): Matplotlib figure for the training plot.
        args (Namespace): Command-line arguments containing parameters.
        rewards_filepath (str): Filepath for saving rewards log (CSV).
        reference_results_filepath (str): Filepath for saving reference results (Pickle).
        plot_training_filepath (str): Filepath for saving plot (PNG).
        plot_eval_filepath (str): Filepath for saving evaluation plot (PNG).
        save_folder (str): Folder to save results.
    """
    flag = "_hw" + str(args.hidden_width) + "_hd" + str(args.hidden_depth) + "_gamma" + str(args.gamma) + "_alpha" + str(args.alpha_d) + "_beta" + str(args.beta_ra) + "_prediction" + str(args.without_prediction)
    agent_type = type(agent).__name__
    if "SAC" in agent_type:
        model_path = "SAC_mdl"
    elif "DDPG" in agent_type:
        model_path = "DDPG_mdl"
    elif "DQN" in agent_type:
        model_path = "DQN_mdl"
    elif "PPO" in agent_type:
        model_path = "PPO_mdl"
    elif "TD3" in agent_type:
        model_path = "TD3_mdl"
    else:
        raise NameError(f"Agent type {agent_type} unrecognized. ")
    if save_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "SAC" in agent_type:
            save_folder = os.path.join(model_path, timestamp + "_SAC" + flag)
        elif "DDPG" in agent_type:
            save_folder = os.path.join(model_path, timestamp + "_DDPG" + flag)
        elif "DQN" in agent_type:
            save_folder = os.path.join(model_path, timestamp + "_DQN" + flag)
        elif "PPO" in agent_type:
            save_folder = os.path.join(model_path, timestamp + "_PPO" + flag)
        elif "TD3" in agent_type:
            save_folder = os.path.join(model_path, timestamp + "_TD3" + flag)
    os.makedirs(save_folder, exist_ok=True)

    # Save training dataset
    env.data_train.to_pickle(os.path.join(save_folder, "dataset_train.pkl"))
    env.data_test.to_pickle(os.path.join(save_folder, "dataset_test.pkl"))

    # Save rewards and training times to a CSV
    training_data = pd.DataFrame({
        'Episode': range(1, len(reward_log) + 1),
        'Total Reward': reward_log,
        'Training Time (s)': time_log,
        'SOC at Departure': soc_log
    })
    training_data.to_csv(os.path.join(save_folder, rewards_filepath), index=False)

    # Save evaluation results
    eval_results_filepath = os.path.join(save_folder, "evaluation_results.pkl")
    eval_data = pd.DataFrame({
        'Episode': [(args.eval_interval * (i + 1)) for i in range(len(eval_log))],
        'Episode Rewards': [r[0] for r in eval_log],
        'Episode final SOCs': [r[1] for r in eval_log]
    })
    eval_data.to_pickle(eval_results_filepath)

    # Save reference results
    reference_results.to_pickle(os.path.join(save_folder, reference_results_filepath))

    # Save final model
    agent.save_networks_replaybuffer(folder=save_folder)
    agent.save_hyperparameters(folder=save_folder)
    if agent.var_hyperparameters:
        var_hyperparameters = pd.DataFrame(agent.var_hyperparameters)
        var_hyperparameters.to_pickle(os.path.join(save_folder, "var_hyperparameters.pkl"))

    # Save plots
    if fig_train is not None:
        fig_train.savefig(os.path.join(save_folder, plot_training_filepath))
    if fig_eval is not None:
        fig_eval.savefig(os.path.join(save_folder, plot_eval_filepath))

    # Save command-line arguments
    params_filepath = os.path.join(save_folder, "training_parameters.txt")
    with open(params_filepath, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Results saved in: {save_folder}")
