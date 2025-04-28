import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import shutil


def initialize_live_plot(var_parameter=None, show_reference=False):
    """
    Initialize the live plotting environment.
    ax1: Episode Reward
    ax2: SOC
    ax3 (optional): variable parameters
    """
    plt.ion()  # Turn on interactive mode for live plotting
    if var_parameter is not None:
        assert type(var_parameter) == str, "var_parameter should be a string"

    if var_parameter is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    lines = []
    axes = []

    ax1.set_title('Live Training Performance')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    line1, = ax1.plot([], [], 'g-', label='Episode Reward')  # Create an empty line for rewards
    lines.append(line1)
    if show_reference:
        line_ref1, = ax1.plot([], [], 'r-', label='MILP Reward')  # Create an empty line for reference rewards MILP
        line_ref2, = ax1.plot([], [], 'm-', label='Uncontrolled Charging Reward')  # Create an empty line for reference rewards Uncontrolled
        lines.append(line_ref1)
        lines.append(line_ref2)
    ax1.grid(True)
    ax1.legend()
    axes.append(ax1)

    ax2.set_title('Final SOC at Departure')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('SOC')
    line2, = ax2.plot([], [], 'b-', label='Final SOC')  # Create an empty line for SOC
    lines.append(line2)

    ax2.grid(True)
    ax2.legend()
    axes.append(ax2)

    if var_parameter:
        ax3.set_title('Variable Parameter ' + var_parameter)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel(var_parameter)
        line3, = ax3.plot([], [], 'r-', label=var_parameter)  # Create an empty line for variable parameters
        ax3.grid(True)
        ax3.legend()
        lines.append(line3)
        axes.append(ax3)
        return fig, axes, lines, var_parameter

    return fig, axes, lines, None


def plot_training_history(history, save_folder):
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss', color='blue')

    # Plot validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, 'Training_history.png'))


def initialize_evaluation_live_plot(show_reference=True):
    """
    Initialize the live plotting environment for evaluation.
    """
    plt.ion()  # Turn on interactive mode for live plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.set_title('Live Evaluation Performance')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Total Reward per cycle')
    line, = ax.plot([], [], 'g-', label='Evaluation Reward')  # Create an empty line for rewards
    if show_reference:
        line_ref1, = ax.plot([], [], 'r-', label='MILP Reward (Best Case)')
        line_ref2, = ax.plot([], [], 'm-', label='Uncontrolled Charging Reward (Reference)')
    lines = [line, line_ref1, line_ref2] if show_reference else [line]
    ax.grid(True)
    ax.legend()
    return fig, ax, lines


def plot_training_process(process_pd, ref_dict=None, aggregate: int = 1, save_folder=None):
    plt.figure(figsize=(12, 6))
    plt.plot(process_pd['Episode'], process_pd['Total Reward'], 'g-', label='Episode Reward Agent')
    if ref_dict is not None and aggregate != 1:
        for each_ref in ref_dict:
            plt.plot(process_pd['Episode'], np.ones(len(process_pd['Episode'])) * each_ref['Total Reward'],
                     label=each_ref['label'])
    plt.xlabel('Cycle') if aggregate != 1 else plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Total Reward per cycle ({aggregate} Episode)') if aggregate != 1 else plt.title('Total Reward per Episode')
    plt.grid()
    plt.legend()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, "total_reward_per_cycle.png")) if aggregate != 1 else plt.savefig(os.path.join(save_folder, "total_reward_per_episode.png"))
    else:
        plt.show(block=False)

    plt.figure(figsize=(12, 6))
    plt.plot(process_pd['Episode'], process_pd['Training Time (s)'], 'b-', label='Training Time')
    plt.xlabel('Cycle') if aggregate != 1 else plt.xlabel('Episode')
    plt.ylabel('Training Time in sec')
    plt.title(f'Training Time per cycle ({aggregate} Episode)') if aggregate != 1 else plt.title('Training Time per Episode')
    plt.grid()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, "training_time_per_cycle.png")) if aggregate != 1 else plt.savefig(os.path.join(save_folder, "training_time_per_episode.png"))
    else:
        plt.show(block=False)


def plot_evaluation_results(eval_list, ref_dict=None, save_folder=None):
    f1 = plt.figure(figsize=(12, 6))
    f2 = plt.figure(figsize=(12, 6))
    for i, eval_data in enumerate(eval_list):
        plt.figure(f1.number)
        plt.plot(eval_data[0], label=f'DRL Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Evaluation Reward per Episode')
        plt.grid(True)

        plt.figure(f2.number)
        plt.plot(eval_data[1], label=f'DRL SOC at Departure')
        plt.xlabel('Episode')
        plt.ylabel('SOC at Departure')
        plt.title('Evaluation final SOC per Episode')
        plt.grid()
        plt.legend()

        if ref_dict:
            plt.figure(f1.number)
            for each_ref in ref_dict:
                plt.plot(each_ref['Reward'], label=each_ref['label'])
                plt.grid(True)
                plt.legend()
    if save_folder is not None:
        f1.savefig(os.path.join(save_folder, "reward_evaluation_testdata.png"))
        f2.savefig(os.path.join(save_folder, "soc_evaluation_testdata.png"))
    else:
        plt.show(block=False)


def plot_operation_process(operation_df, episodes=None, save_folder=None):
    if episodes is None:
        episodes = [0, 50, 100]
    if episodes == "all":
        episodes = range(len(operation_df))
    for each_epi in episodes:
        f1, (ax1_1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
        
        ax1_1.plot(operation_df['Power'][each_epi], 'b-', label='Charging Power')
        ax1_1.set_xlabel('Steps', fontsize=25)
        ax1_1.set_ylabel('Charging Power in kW', color='b', fontsize=25)
        ax1_1.tick_params(axis='y', labelcolor='b', labelsize=20)
        ax1_1.tick_params(axis='x', labelsize=20)
        ax1_1.grid(True)
        ax1_1.legend(fontsize=25)

        ax1_2 = ax1_1.twinx()
        ax1_2.plot(operation_df['Price'][each_epi], 'r-', label='Day-Ahead Price')
        ax1_2.set_ylabel('Price in Euro/MWh', color='r', fontsize=25)
        ax1_2.tick_params(axis='y', labelcolor='r', labelsize=20)

        ax2.plot(operation_df['SOC'][each_epi], 'g-', label='SOC_trajectory')
        ax2.set_xlabel('Steps', fontsize=25)
        ax2.set_ylabel('SOC', fontsize=25)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True)
        ax2.legend(fontsize=25)

        if save_folder is not None:
            f1.savefig(os.path.join(save_folder, f"operation_process_Episode_{each_epi}.png"))
            plt.close(f1)
        else:
            plt.show(block=False)


