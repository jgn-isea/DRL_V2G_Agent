import numpy as np
import pandas as pd
import os
try:
    from env.AgingModel import BatteryAgingModel
    from env.EvseModel import BidirectionalEVCharger
except ImportError:
    from AgingModel import BatteryAgingModel
    from EvseModel import BidirectionalEVCharger


class EVChargingEnvContinuous:
    def __init__(self, forecast_horizon=24, max_timesteps=1000, t_step=0.25, battery_cap=60, power_min=-11,
             power_max=11, alpha_degradation=1, beta_range_anxiety=0.1, price_file="data/electricity_prices_ID1.csv", eval=False):
        """
        Initialize the EV charging environment.
        Parameters:
            forecast_horizon (int): forecast of electricity price in hour.
            max_timesteps (int): Maximum timesteps per episode.
            price_file (str): Path to the CSV file containing electricity prices.
        """
        self.t_step = t_step  # in hour
        self.forecast_horizon = forecast_horizon  # in hour
        self.max_timesteps = max_timesteps
        self.battery_cap = battery_cap
        self.power_min = power_min
        self.power_max = power_max
        self.target_soc = 0.8  # Target SOC for the EV
        self.eta_evse = 0.95
        self.additional_cost = 0.219  # addtional cost for buying electricity from the market (tax, grid fee, etc.)
        # Duration of the charging process, current SOC, Timesteps to next departure， average SOC, accumulated DOD,
        # meanDOD, Energy throughput, Charging cost of the previous SOC step, Prices_forecast
        self.state_size_except_price = 8
        self.state_size = int(forecast_horizon / self.t_step + self.state_size_except_price)
        self.action_size = 1  # Continuous Action Space [-1, 1]
        self.current_step = 0  # starts with t0
        self.battery_cost = self.battery_cap * 100  # Cost of the battery in euro/kWh
        self.AgingModel = BatteryAgingModel(ocv_csv_path='data/ocv_data_35E_dwa_fhi.csv', time_step=self.t_step,
                                            battery_cost=self.battery_cost)  # Parametrization of one cell
        self.Charger = BidirectionalEVCharger(config_file='data/ev_charger_config_Iwafune.json',
                                              rated_power=11)  # EVSE model
        self.beta_range_anxiety = beta_range_anxiety  # Weight for the penalty of range anxiety [euro/kWh^2]
        self.alpha_degradation = alpha_degradation  # Weight for battery degradation cost

        if not eval:
            self._load_price_data(price_file)
            if os.path.exists("data/data_train_seed16_42.pkl") and os.path.exists('data/data_test_seed16_42.pkl'):
                self.data_train = pd.read_pickle("data/data_train_seed16_42.pkl")
                self.data_test = pd.read_pickle("data/data_test_seed16_42.pkl")
            else:
                self.schedule = generate_driving_schedule(2024, 0.25,
                                                          departure_wd_dist={'mean': 7.5, 'std': 1.0, 'min': 6, 'max': 10},
                                                          arrival_wd_dist={'mean': 17.5, 'std': 1.5, 'min': 16, 'max': 20},
                                                          departure_we_dist={'mean': 10.0, 'std': 3.0, 'min': 6, 'max': 14},
                                                          arrival_we_dist={'mean': 19.0, 'std': 3.0, 'min': 15, 'max': 24},
                                                          use_seed=True)  # the driving schedule of the EV
                try:
                    from env.PriceForecastModel import PriceForecastModel
                except ImportError:
                    from PriceForecastModel import PriceForecastModel
                self.ForecastModel = PriceForecastModel(
                    'env/PriceForecastModel/PriceForecast_mdl/BayesianOpt_Weighting_24h_0312_21_07_09/Train_Evaluate_800s/'
                    'lstm_transformer_bys_best_weighting_800s.pth')
                self.ForecastModel.start_eval()
                self.data_train, self.data_test = self._generate_training_and_test_data()
                self.data_train.to_pickle("data/data_train_seed16_42.pkl")
                self.data_test.to_pickle("data/data_test_seed16_42.pkl")

    def load_dataset(self, data_train: pd.DataFrame = None, data_test: pd.DataFrame = None):
        if data_train is not None:
            self.data_train = data_train
        if data_test is not None:
            self.data_test = data_test

    def _generate_training_and_test_data(self, use_seed=True):
        """
        Generate training and testing datasets from driving schedules.

        Parameters:
            use_seed (bool): Whether to use a fixed seed for reproducibility.

        Returns:
            tuple: Two DataFrames (df_train, df_test).
        """
        if use_seed:
            np.random.seed(42)
        # Normalize the price data using 15% quantile and 85% quantile
        q15 = self.price_data.quantile(0.15).values[0]
        q85 = self.price_data.quantile(0.85).values[0]

        # Identify parking events (departures and arrivals)
        a = self.schedule["at_home"] != self.schedule["at_home"].shift()
        events = self.schedule[a].reset_index()

        # Create parking durations DataFrame
        parking_durations = []
        for i in range(0, len(events) - 1, 2):  # Pair departure and arrival
            start_time = events.iloc[i]["index"]
            end_time = events.iloc[i + 1]["index"]
            if self.schedule.loc[start_time, "at_home"] == 1:  # Ensure valid parking event
                parking_durations.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "parking_duration": (end_time - start_time).total_seconds() / 3600
                })
        parking_durations = pd.DataFrame(parking_durations)
        parking_durations = parking_durations.iloc[1:].reset_index(drop=True)  # Drop the first parking event (incomplete

        # Initialize dataframes for weekday and weekend episodes
        df_weekday = pd.DataFrame(columns=["start_time", "end_time", "parking_duration", "price_data"])
        df_weekend = pd.DataFrame(columns=["start_time", "end_time", "parking_duration", "price_data"])

        # Process each parking period
        for _, row in parking_durations.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]
            parking_duration = (end_time - start_time).total_seconds() / 3600

            # Extract price data
            if end_time + pd.Timedelta(hours=self.forecast_horizon) <= self.price_data.index[-1]:
                price_data = self.price_data[start_time:end_time + pd.Timedelta(hours=self.forecast_horizon)].values
            else:
                step = (end_time + pd.Timedelta(hours=self.forecast_horizon) - self.price_data.index[-1]) \
                       / pd.Timedelta(hours=self.t_step)
                price_data = pd.concat([self.price_data[start_time:], self.price_data.iloc[:int(step)]]).values

            # Generate price different forecasting data
            X_data = []
            for idx in range(int(row['parking_duration'] / self.t_step)+1):
                X = self.price_data[start_time + (idx+1)*pd.Timedelta(hours=self.t_step) - pd.Timedelta(hours=7*24): start_time + idx*pd.Timedelta(hours=self.t_step)].values
                X_data.append(X.flatten())
            X_data = np.array(X_data)
            price_forecast = self.ForecastModel.eval_episode(X_data)

            # Normalize price_data and forecast using 15% quantile and 85% quantile
            price_data = price_data.flatten()
            price_data_norm = (price_data - q15) / (q85 - q15)
            price_forecast_norm = (price_forecast - q15) / (q85 - q15)

            episode = {
                "start_time": start_time,
                "end_time": end_time,
                "parking_duration": parking_duration,
                "price_data": price_data,
                "price_data_norm": price_data_norm,
                "price_forecast": price_forecast,
                "price_forecast_norm": price_forecast_norm,
                "initial_soc": np.clip(np.random.normal(0.4, 0.05), 0.3, 0.5)
            }

            # Determine whether it's a weekday or weekend
            if start_time.weekday() < 5:
                if df_weekday.empty:
                    df_weekday = pd.DataFrame([episode])
                else:
                    df_weekday = pd.concat([df_weekday, pd.DataFrame([episode])], ignore_index=True)
            else:
                if df_weekend.empty:
                    df_weekend = pd.DataFrame([episode])
                else:
                    df_weekend = pd.concat([df_weekend, pd.DataFrame([episode])], ignore_index=True)

        # Randomly split into training and testing datasets
        def split_data(df):
            indices = np.arange(len(df))
            np.random.shuffle(indices)
            split_idx = int(len(df) * 2 / 3)
            train_indices, test_indices = indices[:split_idx], indices[split_idx:]
            return df.iloc[train_indices], df.iloc[test_indices]

        df_train_weekday, df_test_weekday = split_data(df_weekday)
        df_train_weekend, df_test_weekend = split_data(df_weekend)

        # Combine weekday and weekend datasets
        df_train = pd.concat([df_train_weekday, df_train_weekend], ignore_index=True)
        df_test = pd.concat([df_test_weekday, df_test_weekend], ignore_index=True)

        return df_train, df_test

    def _load_price_data(self, file_path, start_time="2024-01-01 00:00:00", end_time="2024-12-31 23:45:00",
                                  freq="0.25h", column_name="Value"):
        """
        Load a CSV file and align the data with a specified time range as the index.

        Parameters:
            file_path (str): Path to the CSV file containing the data.
            start_time (str): Start of the time range (e.g., "2023-01-01 00:00:00").
            end_time (str): End of the time range (e.g., "2023-12-31 23:45:00").
            freq (str): Frequency of the time range (e.g., "0.25H" for 15 minutes).
            column_name (str): Name for the data column in the resulting DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the time range as the index and the data as a column.
        """
        # Load the CSV file
        data = pd.read_csv(file_path, header=0)
        data.columns = [column_name]
        data_2023 = pd.read_csv("data/electricity_prices_ID1_2023.csv", header=0)
        data_2023.columns = [column_name]

        # Generate the time range
        start_time = pd.to_datetime(start_time) - pd.Timedelta(hours=7*24)
        time_range = pd.date_range(start=start_time, end=end_time, freq=freq)

        # Verify that the data length matches the time range length
        if len(data)+4*24*7 != len(time_range):
            raise ValueError(
                f"Data length ({len(data)}) does not match time range length ({len(time_range)})."
            )

        # Create the DataFrame with the time range as the index
        data = np.concatenate((data_2023[column_name].values[-24*7*4:], data[column_name].values))
        self.price_data = pd.DataFrame(data, index=time_range, columns=[column_name])

    def reset(self, episode, eval=False, use_prediction=True):
        """
        Reset the environment to its initial state.
        episode: int, the number of episode
        Returns:
            np.ndarray: Initial state of the environment.
        """

        self.current_step = 0
        if not eval:
            self.current_episode = self.data_train.iloc[episode % len(self.data_train)]
        else:
            self.current_episode = self.data_test.iloc[episode % len(self.data_test)]
        self.prices = self.current_episode["price_data"]
        self.prices_norm = self.current_episode["price_data_norm"]
        self.prices_pred_norm = self.current_episode["price_forecast_norm"]
        self.soc = [self.current_episode["initial_soc"]]  # Start with SOC in mid-high range
        self.charging_duration = self.current_episode["parking_duration"] / self.t_step
        self.AgingModel.reset(init_soc=self.current_episode["initial_soc"]*100, charging_duration=self.charging_duration)
        return self.get_state(use_prediction)

    def get_state(self, use_prediction=True):
        """
        Get the current state of the environment.
        Returns:
            np.ndarray: Current state vector.
        """
        time_to_departure = (self.charging_duration - self.current_step)
        if use_prediction:
            price_forecast = np.concatenate((np.array([self.prices_norm[0]]), self.prices_pred_norm[self.current_step][:-1]))   # use real price for the current step
        else:
            price_forecast = self.prices_norm[:int(self.forecast_horizon / self.t_step)]
        return np.concatenate(([self.soc[-1], time_to_departure/100, self.charging_duration/100, np.nanmean(self.soc),
                                self.AgingModel.cum_current_dod/100, self.AgingModel.mean_dod/100, self.AgingModel.current_efc,
                                self.AgingModel.reward_memory/10], price_forecast))

    def step(self, action, use_prediction=True):
        bounds = [self.power_min, self.power_max]  # [lb, ub]
        power_grid = (action[0] + 1) / 2 * (bounds[1] - bounds[0]) + bounds[0]
        # Calculate the power on the EV side and reverse calculate the power on the grid side
        power_cha = np.clip(self.Charger.get_power_ev_side(power_grid),
                            -self.soc[-1] * self.battery_cap / self.t_step,
                            (1 - self.soc[-1]) * self.battery_cap / self.t_step)
        power_grid = self.Charger.get_power_grid_side(power_cha)
        done = False

        # Update SOC
        self.soc.append(self.soc[-1] + power_cha * self.t_step / self.battery_cap)

        # Calculate reward
        # Trading cost/revenue: exemption of the additional cost if the energy will be sold on the market
        reward = -power_grid * self.t_step * (self.prices[0] / 1000 + self.additional_cost)
        # Battery aging cost
        reward += self.alpha_degradation * self.AgingModel.calc_immediate_reward(np.array(self.soc) * 100, self.charging_duration)

        # Check if car departs
        if self.current_step == self.charging_duration - 1:
            # Sparse reward term: Penalty for Range Anxiety
            reward -= self.calc_range_anxiety_penalty()
            done = True

        # Transition prices
        self.prices = self.prices[1:]
        self.prices_norm = self.prices_norm[1:]

        # Update step count
        self.current_step += 1

        if self.current_step >= self.max_timesteps:
            done = True

        return self.get_state(use_prediction), reward, done, self.soc[-1], power_grid

    def calc_range_anxiety_penalty(self):
        """
        Calculate the penalty for range anxiety based on the current SOC, the target SOC, the maximum charging power,
        and the weight alpha.
        Returns:
            float: Penalty for range anxiety.
        """
        if self.current_step == (self.charging_duration - 1) and self.soc[-1] < self.target_soc:
            return ((self.target_soc - self.soc[-1]) * self.battery_cap) ** 2 * self.beta_range_anxiety
        else:
            return 0

    def get_mask(self):
        """
        Get the mask for the current state, which indicates the valid action space.
        :return:
        """
        mask = np.zeros(self.state_size)
        mask[:self.state_size_except_price+int(self.charging_duration-self.current_step)] = 1
        return mask


class EVChargingEnvDiscrete(EVChargingEnvContinuous):
    def __init__(self, forecast_horizon=24, max_timesteps=1000, t_step=0.25, battery_cap=60, power_min=-11,
                 power_max=11, alpha_degradation=1, beta_range_anxiety=0.1, price_file="data/electricity_prices_ID1.csv", eval=False):
        """
        Initialize the EV charging environment.
        Parameters:
            forecast_horizon (int): Number of timesteps for electricity price forecasts.
            max_timesteps (int): Maximum timesteps per episode.
            price_file (str): Path to the CSV file containing electricity prices.
        """
        super().__init__(forecast_horizon, max_timesteps, t_step, battery_cap, power_min, power_max, alpha_degradation, beta_range_anxiety, price_file, eval)
        self.action_space = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        self.action_size = len(self.action_space)  # discrete action space: Charging/discharging levels [-22, -11, 0, 11, 22]

    def step(self, action, use_prediction=True):
        split = self.action_space[action]
        bounds = [self.power_min, self.power_max]  # [lb, ub]
        power_grid = (split + 1) / 2 * (bounds[1] - bounds[0]) + bounds[0]
        # Calculate the power on the EV side and reverse calculate the power on the grid side
        power_cha = np.clip(self.Charger.get_power_ev_side(power_grid),
                            -self.soc[-1] * self.battery_cap / self.t_step,
                            (1 - self.soc[-1]) * self.battery_cap / self.t_step)
        power_grid = self.Charger.get_power_grid_side(power_cha)
        done = False

        # Update SOC
        self.soc.append(self.soc[-1] + power_cha * self.t_step / self.battery_cap)

        # Calculate reward
        # Trading cost/revenue: exemption of the additional cost if the energy will be sold on the market
        reward = -power_grid * self.t_step * (self.prices[0] / 1000 + self.additional_cost)
        # Battery aging cost
        reward += self.alpha_degradation * self.AgingModel.calc_immediate_reward(np.array(self.soc) * 100, self.charging_duration)

        # Check if car departs
        if self.current_step == self.charging_duration - 1:
            # Sparse reward term: Penalty for Range Anxiety
            reward -= self.calc_range_anxiety_penalty()
            done = True

        # Transition prices
        self.prices = self.prices[1:]
        self.prices_norm = self.prices_norm[1:]

        # Update step count
        self.current_step += 1

        if self.current_step >= self.max_timesteps:
            done = True

        return self.get_state(use_prediction), reward, done, self.soc[-1], power_grid


def generate_driving_schedule(year, t_step, departure_wd_dist, arrival_wd_dist, departure_we_dist, arrival_we_dist,
                              use_seed=True):
    """
    Generate a driving schedule for an EV over a year.

    Parameters:
        year (int): The year for which the schedule is generated (e.g., 2023).
        t_step (float): Time step in hours (e.g., 0.25 for 15 minutes).
        departure_wd_dist (dict): Normal distribution parameters for departure times on weekdays (e.g., {'mean': 7.5, 'std': 1.0, 'min': 0, 'max': 24}).
        arrival_wd_dist (dict): Normal distribution parameters for arrival times on weekdays (e.g., {'mean': 17.5, 'std': 1.5, 'min': 0, 'max': 24}).
        departure_we_dist (dict): Normal distribution parameters for departure times on weekends (e.g., {'mean': 9.0, 'std': 2.0, 'min': 0, 'max': 24}).
        arrival_we_dist (dict): Normal distribution parameters for arrival times on weekends (e.g., {'mean': 19.0, 'std': 2.0, 'min': 0, 'max': 24}).
        use_seed (bool, optional): Seed for reproducibility of random departure and arrival times. Default is True.

    Returns:
        pd.DataFrame: DataFrame with a datetime index and a column indicating whether the EV is at home (1) or not (0).
    """
    if use_seed:
        np.random.seed(16)

    # Generate date range for the year
    start_date = f"{year}-01-01 00:00:00"
    end_date = f"{year}-12-31 23:59:00"
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Initialize arrays for schedule
    schedule = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq=f"{t_step}h"), columns=["at_home"])
    schedule["at_home"] = 1

    # Iterate over the days in the year
    for date in date_range:
        # Determine if it's a weekday or weekend
        is_weekday = date.weekday() < 5

        # Generate departure and arrival times based on the distribution
        if is_weekday:
            departure_time = np.clip(
                np.random.normal(loc=departure_wd_dist['mean'], scale=departure_wd_dist['std']),
                departure_wd_dist['min'], departure_wd_dist['max']
            )
            arrival_time = np.clip(
                np.random.normal(loc=arrival_wd_dist['mean'], scale=arrival_wd_dist['std']),
                arrival_wd_dist['min'], arrival_wd_dist['max']
            )
        else:
            departure_time = np.clip(
                np.random.normal(loc=departure_we_dist['mean'], scale=departure_we_dist['std']),
                departure_we_dist['min'], departure_we_dist['max']
            )
            arrival_time = np.clip(
                np.random.normal(loc=arrival_we_dist['mean'], scale=arrival_we_dist['std']),
                arrival_we_dist['min'], arrival_we_dist['max']
            )

        # Round to nearest timestep
        departure_time = round(departure_time / t_step) * t_step
        arrival_time = round(arrival_time / t_step) * t_step

        # Ensure arrival is after departure
        if arrival_time <= departure_time:
            arrival_time += 24

        # Mark the EV as not at home between departure and arrival
        departure_idx = date + pd.Timedelta(hours=departure_time)
        arrival_idx = date + pd.Timedelta(hours=arrival_time-t_step)

        schedule.loc[departure_idx:arrival_idx, "at_home"] = 0

    return schedule


if __name__ == "__main__":
    env = EVChargingEnvContinuous(price_file="./data/electricity_prices_ID1.csv")
