# -*- coding = utf-8 -*-
# @Time: 04.01.2025 23:28
# @Author: J.Gong
# @File: decorator
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
from functools import wraps


def live_plot(func):
    """
    Decorator to handle live plotting during training.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('plot_elements')[-1] == "train_plot":  # Training plot
            # Extract plotting arguments
            fig, axes, lines, var_parameter, _ = kwargs.get('plot_elements')
            if len(axes) == 2:
                # ax1: Episode Reward, ax2: SOC
                ax1, ax2 = axes
            elif len(axes) == 3:
                # ax1: Episode Reward, ax2: SOC, ax3: variable parameter
                ax1, ax2, ax3 = axes
            else:
                raise ValueError("Invalid number of axes provided.")

            # Call the original function
            result = func(*args, **kwargs)

            # Extract logs
            reward_log = kwargs.get('reward_log')
            soc_log = kwargs.get('soc_log')

            # Update the plots
            if len(lines) == 2:
                line1, line2 = lines
                line1.set_data(range(len(reward_log)), reward_log)
                line2.set_data(range(len(soc_log)), soc_log)
            elif len(lines) == 3:
                if var_parameter in args[0].var_hyperparameters.keys():
                    var_parameter_log = args[0].var_hyperparameters[var_parameter]
                else:
                    var_parameter_log = None
                line1, line2, line3 = lines
                line1.set_data(range(len(reward_log)), reward_log)
                line2.set_data(range(len(soc_log)), soc_log)
                line3.set_data(range(len(var_parameter_log)), var_parameter_log)
            elif len(lines) == 4:
                reference_values = kwargs.get('reference_values')
                line1, line2, line_ref1, line_ref2 = lines
                line1.set_data(range(len(reward_log)), reward_log)
                line2.set_data(range(len(soc_log)), soc_log)
                line_ref1.set_data(range(len(reward_log)), np.ones(len(reward_log)) * reference_values["MILP Train"])
                line_ref2.set_data(range(len(reward_log)), np.ones(len(reward_log)) * reference_values["Uncontrolled Train"])
            elif len(lines) == 5:
                alpha_log = [_.numpy() for _ in args[0].var_hyperparameters["alpha"]]
                reference_values = kwargs.get('reference_values')
                line1, line_ref1, line_ref2, line2, line3 = lines
                line1.set_data(range(len(reward_log)), reward_log)
                line2.set_data(range(len(soc_log)), soc_log)
                line3.set_data(range(len(alpha_log)), alpha_log)
                line_ref1.set_data(range(len(reward_log)), np.ones(len(reward_log)) * reference_values["MILP Train"])
                line_ref2.set_data(range(len(reward_log)), np.ones(len(reward_log)) * reference_values["Uncontrolled Train"])
            else:
                raise ValueError("Invalid number of lines provided.")
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
            if len(axes) == 3:
                ax3.relim()
                ax3.autoscale_view()
            plt.pause(0.01)

        elif kwargs.get('plot_elements')[-1] == "eval_plot":  # Evaluation plot
            # Extract plotting arguments
            fig, ax, lines, _ = kwargs.get('plot_elements')

            # Call the original function
            result = func(*args, **kwargs)

            # Extract logs
            reward_eval_log = kwargs.get('eval_log')
            reward_eval_log = [np.sum(reward[0]) for reward in reward_eval_log]

            # Update the plot
            if len(lines) == 1:
                line = lines[0]
                line.set_data(range(len(reward_eval_log)), reward_eval_log)
            elif len(lines) == 3:
                reference_values = kwargs.get('reference_values')
                line, line_ref1, line_ref2 = lines
                line.set_data(range(len(reward_eval_log)), reward_eval_log)
                line_ref1.set_data(range(len(reward_eval_log)), np.ones(len(reward_eval_log)) * reference_values["MILP Test"])
                line_ref2.set_data(range(len(reward_eval_log)), np.ones(len(reward_eval_log)) * reference_values["Uncontrolled Test"])
            else:
                raise ValueError("Invalid number of lines provided.")
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        elif kwargs.get('plot_elements')[-1] == "no_plot":  # deactivate plot
            # Call the original function
            result = func(*args, **kwargs)

        else:
            raise ValueError("Invalid number of plot elements provided.")


        return result

    return wrapper

