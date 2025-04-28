from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.AgingModel import BatteryAgingModel
from env.EvseModel import BidirectionalEVCharger


# MILP model for EV charging optimization
def create_milp_ev_charging_model(dataset: pd.Series, alpha_aging_cost, soc_target, soc_init, alpha_degradation, beta_range_anxiety, t_step=0.25, battery_capacity=60, max_power=11,
                                  eta_evse=0.9, additional_cost=0.219, use_prediction=False, price_prediction=None, n_step=None):
    # Load data from dataset
    if not use_prediction:
        prices = dataset["price_data"]
        t_start = dataset["start_time"]
        t_end = dataset["end_time"]
        num_step = (t_end - t_start) / pd.Timedelta(hours=t_step)
    else:
        prices = price_prediction
        num_step = n_step

    model = ConcreteModel()

    # Sets and parameters
    model.T = RangeSet(1, num_step)   # Timesteps
    model.P = Param(model.T, initialize=lambda model, t: prices[t - 1], within=Reals)  # Electricity prices (allow negative values)

    # Variables
    model.soc = Var(model.T, bounds=(0, 1))  # State of charge (0 to 1)
    model.charging_power = Var(model.T, bounds=(0, max_power))  # Charging power (kW)
    model.discharging_power = Var(model.T, bounds=(0, max_power))  # Discharging power (kW)
    model.is_charging = Var(model.T, within=Binary)  # Binary variable for discharging (0) or charging (1)
    model.aging_cost = Var(bounds=(None, None))  # Aging cost
    model.electricity_cost = Var(bounds=(None, None))  # Total Electricity cost/reward
    model.soc_dep = Var(bounds=(None, None))  # SOC at departure
    model.shortfall = Var(bounds=(0, 1))  # SOC shortfall

    # Constraints
    def mutual_exclusivity_rule_1(model, t):
        # Ensure charging power is 0 when not charging
        return model.charging_power[t] <= model.is_charging[t] * max_power
    model.mutual_exclusivity_cons_1 = Constraint(model.T, rule=mutual_exclusivity_rule_1)

    def mutual_exclusivity_rule_2(model, t):
        # Ensure discharging power is 0 when charging
        return model.discharging_power[t] <= (1 - model.is_charging[t]) * max_power
    model.mutual_exclusivity_cons_2 = Constraint(model.T, rule=mutual_exclusivity_rule_2)

    def soc_update_rule(model, t):
        if t == 1:
            return Constraint.Skip  # Skip the first timestep and when EV is not at home
        return model.soc[t] == model.soc[t - 1] + (model.charging_power[t - 1] * eta_evse - model.discharging_power[t - 1] / eta_evse) * t_step / battery_capacity
    model.soc_update_cons = Constraint(model.T, rule=soc_update_rule)

    def soc_initial_rule(model):
        return model.soc[1] == soc_init  # Initial SOC is 30%, align with EVChargingEnv
    model.soc_initial_cons = Constraint(rule=soc_initial_rule)

    def soc_departure_rule(model, t):
        if t == num_step:
            return model.soc[t] + (model.charging_power[t] * eta_evse - model.discharging_power[t] / eta_evse) * t_step / battery_capacity == model.soc_dep  # SOC at departure
        return Constraint.Skip
    model.soc_departure = Constraint(model.T, rule=soc_departure_rule)

    def shortfall_lower_bound_rule(model):
        return model.shortfall >= soc_target - model.soc_dep
    model.shortfall_cons = Constraint(rule=shortfall_lower_bound_rule)

    def aging_cost_rule(model):
        return model.aging_cost == sum((alpha_aging_cost * (model.charging_power[t] + model.discharging_power[t]) * t_step) for t in model.T)
    model.aging_cost_cons = Constraint(rule=aging_cost_rule)

    def electricity_cost_rule(model):
        # Electricity cost/revenue is the sum of trading cost in each step and additional charges
        return model.electricity_cost == sum(((model.P[t] / 1000 + additional_cost) * (model.charging_power[t] - model.discharging_power[t]) * t_step) for t in model.T)
    model.electricity_cost_cons = Constraint(rule=electricity_cost_rule)

    # Objective function: Minimize electricity cost/reward, aging cost, and range anxiety penalty
    def objective_rule(model):
        return model.aging_cost * alpha_degradation + model.electricity_cost + beta_range_anxiety * (model.shortfall * battery_capacity) ** 2
    model.obj = Objective(rule=objective_rule, sense=minimize)

    return model


def milp_evaluate_dataset(dataset: pd.DataFrame, soc_target=0.8, battery_cost=6000, t_step=0.25, battery_capacity=60, max_power=11,
                          eta_evse=0.9, additional_cost=0.219, alpha_degradation=1, beta_range_anxiety=0.1, verbose: int = 0):
    """
    Evaluate the MILP model on a dataset of EV charging episodes.

    :param dataset: DataFrame containing the dataset of EV charging episodes.
    :param soc_init: Initial state of charge (SOC) of the battery, default is 0.3.
    :param t_step: Time step for the simulation, default is 0.25 hours.
    :param battery_capacity: Capacity of the battery in kWh, default is 100 kWh.
    :param max_power: Maximum charging/discharging power in kW, default is 22 kW.
    :param verbose: Verbosity level (0, 1, or 2), default is 0.
    :return: Tuple containing rewards_milp, total_reward_milp, average_reward, and action_milp.
    """
    if verbose not in [0, 1, 2]:
        raise ValueError("verbose must be 0, 1, or 2")
    num_episodes = len(dataset)
    rewards_milp = []
    alpha_aging_cost_log = []
    total_reward_milp = 0
    action_milp = []
    soc_milp = []
    aging_model = BatteryAgingModel(ocv_csv_path='data/ocv_data_35E_dwa_fhi.csv', time_step=t_step, battery_cost=battery_cost)
    Charger = BidirectionalEVCharger(config_file='data/ev_charger_config_Iwafune.json', rated_power=11)  # EVSE model
    alpha_aging_cost = 0.1  # euro/kWh
    for idx, e in dataset.iterrows():
        action_episode = []
        soc_init = e["initial_soc"]
        while True:
            milp_mdl = create_milp_ev_charging_model(e, alpha_aging_cost, soc_target, soc_init, alpha_degradation, beta_range_anxiety, t_step, battery_capacity, max_power, eta_evse, additional_cost)
            solver = SolverFactory('gurobi')
            results = solver.solve(milp_mdl, tee=True)

            soc = np.array([milp_mdl.soc[t].value for t in milp_mdl.T] + [milp_mdl.soc_dep.value]) * 100
            aging_cost = -aging_model.calc_episode_reward(soc, charging_duration=len(milp_mdl.T))
            kWh_throughput = sum((milp_mdl.charging_power[t].value + milp_mdl.discharging_power[t].value) * t_step for t in milp_mdl.T)
            alpha_aging_cost_new = aging_cost / kWh_throughput if kWh_throughput > 0 else 10

            # Display results of this episode
            if verbose == 2:
                print("\nOptimal Solution:")
                for t in milp_mdl.T:
                    print(f"Time {t}: SOC = {value(milp_mdl.soc[t]):.2f}, Price = {value(milp_mdl.P[t]):.2f}, "
                          f"Charging Power = {value(milp_mdl.charging_power[t]):.2f}, Discharging Power = {value(milp_mdl.discharging_power[t]):.2f}")
            if abs(alpha_aging_cost_new - alpha_aging_cost) < 0.001:
                break
            else:
                alpha_aging_cost = alpha_aging_cost_new

        # Collect charging/discharging actions
        for t in range(1, len(milp_mdl.T) + 1):
            if milp_mdl.is_charging[t].value == 0:
                action_episode.append(-milp_mdl.discharging_power[t].value)
            else:
                action_episode.append(milp_mdl.charging_power[t].value)

        # Calculate the real soc with charger efficiency and save it
        soc_real = [soc[0]]
        for i in range(len(action_episode)):
            power_dc = Charger.get_power_ev_side(action_episode[i])
            soc_real.append((soc_real[-1] / 100 + power_dc * t_step / battery_capacity) * 100)
            soc_real[-1] = np.clip(soc_real[-1], 0, 100)
            action_real = Charger.get_power_grid_side((soc_real[-1] - soc_real[-2]) / 100 * battery_capacity / t_step)
            action_episode[i] = action_real

        soc_real = np.array(soc_real)
        action_episode = np.array(action_episode)
        action_milp.append(action_episode)
        soc_milp.append(soc_real)
        alpha_aging_cost_log.append(alpha_aging_cost)
        # Calculate total reward (electricity cost/reward und aging cost/reward and range anxiety penalty)
        aging_reward = aging_model.calc_episode_reward(soc, charging_duration=len(milp_mdl.T))
        range_anxiety_penalty = beta_range_anxiety * (milp_mdl.shortfall.value * battery_capacity) ** 2
        episode_reward = alpha_degradation * aging_reward - milp_mdl.electricity_cost.value - range_anxiety_penalty
        rewards_milp.append(episode_reward)
        total_reward_milp += episode_reward

    return rewards_milp, total_reward_milp, action_milp, alpha_aging_cost_log, soc_milp


def mpc_evaluate_dataset(dataset: pd.DataFrame, soc_target=0.8, battery_cost=6000, t_step=0.25, battery_capacity=60, max_power=11,
                          eta_evse=0.9, additional_cost=0.219, alpha_degradation=1, beta_range_anxiety=0.1):
    """
    Evaluate the MPC model on a dataset of EV charging episodes.

    :param dataset: DataFrame containing the dataset of EV charging episodes.
    :param soc_init: Initial state of charge (SOC) of the battery, default is 0.3.
    :param t_step: Time step for the simulation, default is 0.25 hours.
    :param battery_capacity: Capacity of the battery in kWh, default is 100 kWh.
    :param max_power: Maximum charging/discharging power in kW, default is 22 kW.
    :param verbose: Verbosity level (0, 1, or 2), default is 0.
    :return: Tuple containing rewards_milp, total_reward_milp, average_reward, and action_milp.
    """
    num_episodes = len(dataset)
    Charger = BidirectionalEVCharger(config_file='data/ev_charger_config_Iwafune.json', rated_power=11)  # EVSE model
    rewards_mpc = []
    alpha_aging_cost_log = []
    total_reward_mpc = 0
    action_mpc = []
    soc_mpc = []
    aging_model = BatteryAgingModel(ocv_csv_path='data/ocv_data_35E_dwa_fhi.csv', time_step=t_step, battery_cost=battery_cost)
    alpha_aging_cost = 0.15  # euro/kWh
    for idx, e in dataset.iterrows():
        action_epi = []
        soc_epi = [e["initial_soc"]*100]
        charging_duration = (e["end_time"] - e["start_time"]) / pd.Timedelta(hours=t_step)
        for step in range(int(charging_duration)):
            price_pred = [e["price_data"][0]] + e["price_forecast"][step][:-1]
            n_step = charging_duration - step
            soc_init = soc[1]/100 if step != 0 else e["initial_soc"]
            milp_mdl = create_milp_ev_charging_model(e, alpha_aging_cost, soc_target, soc_init, alpha_degradation, beta_range_anxiety, t_step, battery_capacity, max_power, eta_evse, additional_cost,
                                                     use_prediction=True, price_prediction=price_pred, n_step=n_step)
            solver = SolverFactory('gurobi')
            results = solver.solve(milp_mdl, tee=True)

            soc = np.array([milp_mdl.soc[t].value for t in milp_mdl.T] + [milp_mdl.soc_dep.value]) * 100
            # get first action
            if milp_mdl.is_charging[milp_mdl.T[1]].value == 0:
                first_action = -milp_mdl.discharging_power[milp_mdl.T[1]].value
            else:
                first_action = milp_mdl.charging_power[milp_mdl.T[1]].value
            # Calculate the real soc with charger efficiency and save it
            power_dc = Charger.get_power_ev_side(first_action)
            soc_real = (soc[0] / 100 + power_dc * t_step / battery_capacity) * 100
            soc_real = np.clip(soc_real, 0, 100)
            action_real = Charger.get_power_grid_side((soc_real - soc[0]) / 100 * battery_capacity / t_step)
            action_epi.append(action_real)
            soc[1] = soc_real
            soc_epi.append(soc_real)

        soc_epi = np.array(soc_epi)
        action_epi = np.array(action_epi)

        action_mpc.append(action_epi)
        soc_mpc.append(soc_epi)
        alpha_aging_cost_log.append(alpha_aging_cost)
        # Calculate total reward (electricity cost/reward und aging cost/reward and range anxiety penalty)
        aging_reward = aging_model.calc_episode_reward(soc_epi, charging_duration=charging_duration)
        range_anxiety_penalty = beta_range_anxiety * (milp_mdl.shortfall.value * battery_capacity) ** 2
        episode_reward = alpha_degradation * aging_reward - milp_mdl.electricity_cost.value - range_anxiety_penalty
        rewards_mpc.append(episode_reward)
        total_reward_mpc += episode_reward

    # Compute average reward and other statistics
    average_reward = total_reward_mpc / num_episodes
    print(f"Total Reward of MILP over Test Dataset: {total_reward_mpc}")
    print(f"Average Reward of MILP per Episode: {average_reward}")

    return rewards_mpc, total_reward_mpc, action_mpc, alpha_aging_cost_log, soc_mpc


# Rule-based uncontrolled charging simulation
def simulate_uncontrolled_charging(dataset: pd.Series, aging_model, battery_cost, charger, soc_target, alpha_degradation, beta_range_anxiety,
                                   soc_init, t_step=0.25, battery_capacity=60, additional_cost=0.219, save=False):
    """
    Simulate uncontrolled charging for a single EV charging episode.

    :param dataset: Series containing the dataset of EV charging episode.
    :param soc_init: Initial state of charge (SOC) of the battery, default is 0.3.
    :param t_step: Time step for the simulation, default is 0.25 hours.
    :param battery_capacity: Capacity of the battery in kWh, default is 100 kWh.
    :param max_power: Maximum charging/discharging power in kW, default is 22 kW.
    :param save: Boolean indicating whether to save the results to an Excel file, default is False.
    :return: Tuple containing the total reward and the charging power array.
    """
    # Load data from dataset
    prices = dataset["price_data"]
    t_start = dataset["start_time"]
    t_end = dataset["end_time"]
    num_step = (t_end - t_start) / pd.Timedelta(hours=t_step)
    if num_step - int(num_step) != 0:
        raise ValueError("num_step must be an integer number!")
    num_step = int(num_step)

    # Initialize SOC and charging power
    soc = np.ones(num_step + 1) * soc_init  # Initial SOC
    power_grid = np.zeros(num_step)
    power_cha = np.zeros(num_step)

    # Simulate uncontrolled charging
    for t in range(num_step):
        if soc[t] < soc_target:
            power_cha[t] = min(charger.P_ev_max_cha, (soc_target - soc[t]) * battery_capacity / t_step)
            power_grid[t] = charger.get_power_grid_side(power_cha[t])
            new_soc = soc[t] + power_cha[t] * t_step / battery_capacity
            soc[t+1:] = new_soc

    # Calculate total reward (electricity cost/reward)
    reward = sum((prices[t]/1000 + additional_cost) * power_grid[t] * t_step for t in range(num_step)) * -1
    # Calculate aging cost
    aging_cost = -alpha_degradation * aging_model.calc_episode_reward(soc*100, charging_duration=num_step)
    reward -= aging_cost
    alpha_aging_cost = aging_cost / (power_grid.sum() * t_step)
    # Calculate penalty for range anxiety
    if soc[-1] < soc_target:
        reward -= ((soc_target - soc[-1]) * battery_capacity) ** 2 * beta_range_anxiety

    if save:
        # Prepare results for export
        results_data = []
        for t in range(1, num_step):
            results_data.append({
                "Time": t,
                "SOC": soc[t],
                "GridPower": power_grid[t],
                "ChargingPower": power_cha[t],
                "Price": prices[t]
            })

        # Convert results to a DataFrame and save to Excel
        results_df = pd.DataFrame(results_data)
        results_df.to_excel("uncontrolled_charging_results.xlsx", index=False)

        # Display results
        print("\nUncontrolled Charging Solution:")
        for t in range(1, num_step + 1):
            print(f"Time {t}: SOC = {soc[t]:.2f}, Grid Power = {power_grid[t]:.2f}, Charging Power = {power_cha[t]:.2f}, Price = {prices[t]:.2f}")

        print(f"\nTotal Reward (Electricity Cost/Reward): {reward:.2f}")

    return reward, power_grid, alpha_aging_cost, soc


def uncontrolled_evaluate_dataset(dataset: pd.DataFrame, battery_cost, charger, soc_target, alpha_degradation, beta_range_anxiety, t_step=0.25, battery_capacity=60,
                                  verbose: int = 0):
    """
    Evaluate the uncontrolled charging policy on a dataset of EV charging episodes.

    :param dataset: DataFrame containing the dataset of EV charging episodes.
    :param soc_init: Initial state of charge (SOC) of the battery, default is 0.3.
    :param t_step: Time step for the simulation, default is 0.25 hours.
    :param battery_capacity: Capacity of the battery in kWh, default is 100 kWh.
    :param max_power: Maximum charging/discharging power in kW, default is 22 kW.
    :param verbose: Verbosity level (0, 1, or 2), default is 0.
    :return: Tuple containing rewards_uncontrolled, total_reward_uncontrolled, average_reward, and action_uncontrolled.
    """
    if verbose not in [0, 1, 2]:
        raise ValueError("verbose must be 0, 1, 2.")
    aging_model = BatteryAgingModel(ocv_csv_path='data/ocv_data_35E_dwa_fhi.csv', time_step=t_step, battery_cost=battery_cost)
    num_episodes = len(dataset)
    rewards_uncontrolled = []
    alpha_aging_cost_log = []
    total_reward_uncontrolled = 0
    action_uncontrolled = []
    soc_uncontrolled = []
    for idx, e in dataset.iterrows():
        soc_init = e["initial_soc"]
        episode_reward, action_episode, alpha_aging_cost, soc = simulate_uncontrolled_charging(e, aging_model, battery_cost, charger, soc_target, alpha_degradation, beta_range_anxiety, soc_init, t_step, battery_capacity)
        rewards_uncontrolled.append(episode_reward)
        total_reward_uncontrolled += episode_reward
        action_uncontrolled.append(action_episode)
        alpha_aging_cost_log.append(alpha_aging_cost)
        soc_uncontrolled.append(soc)
        if verbose == 2:
            print(f"Episode {idx + 1}/{num_episodes}: Reward = {episode_reward}")

    # Compute average reward and other statistics
    average_reward = total_reward_uncontrolled / num_episodes
    if verbose == 1 or verbose == 2:
        print(f"Total Reward of Uncontrolled Charging over Test Dataset: {total_reward_uncontrolled}")
        print(f"Average Reward of Uncontrolled Charging per Episode: {average_reward}")

    return rewards_uncontrolled, total_reward_uncontrolled, action_uncontrolled, alpha_aging_cost_log, soc_uncontrolled


def get_reference_results(env):
    rewards_uncontrolled_train, total_reward_uncontrolled_train, action_uncontrolled_train, alpha_aging_cost_uncontrolled_train, _ = uncontrolled_evaluate_dataset(
        env.data_train, env.battery_cost, env.Charger, env.target_soc, env.alpha_degradation, env.beta_range_anxiety,
        battery_capacity=env.battery_cap, verbose=1)
    rewards_uncontrolled_test, total_reward_uncontrolled_test, action_uncontrolled_test, alpha_aging_cost_uncontrolled_test, _ = uncontrolled_evaluate_dataset(
        env.data_test, env.battery_cost, env.Charger, env.target_soc, env.alpha_degradation, env.beta_range_anxiety,
        battery_capacity=env.battery_cap, verbose=1)
    rewards_milp_train, total_reward_milp_train, action_milp_train, alpha_aging_cost_milp_train, _ = milp_evaluate_dataset(
        dataset=env.data_train, soc_target=env.target_soc, battery_cost=env.battery_cost, t_step=env.t_step,
        battery_capacity=env.battery_cap, max_power=env.power_max, additional_cost=env.additional_cost,
        alpha_degradation=env.alpha_degradation, beta_range_anxiety=env.beta_range_anxiety, verbose=1)
    rewards_milp_test, total_reward_milp_test, action_milp_test, alpha_aging_cost_milp_test, _ = milp_evaluate_dataset(
        dataset=env.data_test, soc_target=env.target_soc, battery_cost=env.battery_cost, t_step=env.t_step,
        battery_capacity=env.battery_cap, max_power=env.power_max, additional_cost=env.additional_cost, verbose=1)

    total_reward_results = {
        "MILP Train": total_reward_milp_train,
        "MILP Test": total_reward_milp_test,
        "Uncontrolled Train": total_reward_uncontrolled_train,
        "Uncontrolled Test": total_reward_uncontrolled_test
    }
    reference_results = pd.DataFrame(
        {"MILP train": [rewards_milp_train, action_milp_train, total_reward_milp_train, alpha_aging_cost_milp_train],
         "MILP test": [rewards_milp_test, action_milp_test, total_reward_milp_test, alpha_aging_cost_milp_test],
         "Uncontrolled train": [rewards_uncontrolled_train, action_uncontrolled_train,
                                total_reward_uncontrolled_train, alpha_aging_cost_uncontrolled_train],
         "Uncontrolled test": [rewards_uncontrolled_test, action_uncontrolled_test,
                               total_reward_uncontrolled_test, alpha_aging_cost_uncontrolled_test]},
        index=["rewards", "actions", "total_reward", "alpha_aging_cost"])
    return reference_results, total_reward_results


if __name__ == "__main__":
    from env.EVChargingEnv import EVChargingEnvContinuous as EVChargingEnv
    env = EVChargingEnv()
    rewards_mpc, total_reward_mpc, action_mpc, alpha_aging_cost_log, soc_mpc = mpc_evaluate_dataset(env.data_train, env.target_soc, env.battery_cost, env.t_step, env.battery_cap, env.power_max, env.eta_evse, env.additional_cost, env.alpha_degradation, env.beta_range_anxiety)
    print(f"Total Reward of MPC over Test Dataset: {total_reward_mpc}")
    print(f"Average Reward of MPC per Episode: {total_reward_mpc / len(env.data_test)}")

