import numpy as np
import pandas as pd
import rainflow


class BatteryAgingModel:
    def __init__(self, ocv_csv_path, time_step, battery_cost, EOL=0.8, capacity=3.6, calendar_params=None, cyclic_params=None):
        """
        Initialize the Battery Aging Model with default or user-provided parameters.
        :param ocv_csv_path: Path to the OCV data file (CSV format).
        :param capacity: Battery capacity in Ah.
        :param calendar_params: Dictionary for calendar aging parameters (a, b, c).
        :param cyclic_params: Dictionary for cyclic aging parameters (a, b, c, minv).
        :param t_factor: Time dependency exponent for calendar aging.
        :param q_factor: Ampere-hour dependency exponent for cyclic aging.
        """
        self.ocv_data = self.load_ocv_data(ocv_csv_path)
        self.cap = capacity
        self.calendar_params = calendar_params or {
            "a": 7.543e6,
            "b": -23.75e6,
            "c": -6976
        }
        self.cyclic_params = cyclic_params or {
            "a": 1.233e-4,
            "b": -9.729,
            "c": -0.1488,
            "minv": -287.6
        }
        self.t_factor = 0.8
        self.q_factor = 0.5
        self.time_step = time_step
        self.battery_cost = battery_cost
        self.EOL = EOL

    def reset(self, init_soc, charging_duration):
        soc = np.array([init_soc])
        self.reward_memory = self.calc_episode_reward(soc, charging_duration)
        self.cum_current_dod = 0
        self.mean_dod = 0
        self.current_efc = 0

    def load_ocv_data(self, path):
        """
        Load OCV data from a CSV file.
        :param path: Path to the CSV file.
        :return: Loaded OCV data as a DataFrame.
        """
        return pd.read_csv(path)

    def calculate_voltage_dod(self, SOC):
        """
        Estimate voltage and calculate Depth of Discharge (DOD) from SOC.
        :param SOC: State of Charge array.
        :return: Voltage and DOD arrays.
        """
        voltage_map = self.ocv_data[['SOC', 'Voltage']]
        voltage_map = voltage_map.drop_duplicates().sort_values(by='SOC')

        # Interpolate voltage based on SOC
        voltage = np.interp(np.nanmean(SOC), voltage_map['SOC'].values, voltage_map['Voltage'].values)

        # Calculate DOD with the rainflow counting algorithm
        dod = self.count_dod(SOC)

        return voltage, dod

    def calculate_ampere_hour_throughput(self, current_profile, time_step):
        """
        Calculate the Ampere-hour throughput from the current profile.
        :param current_profile: Array of current values (in Amperes).
        :param time_step: Time step duration (in hours).
        :return: Ampere-hour throughput array.
        """
        return np.sum(abs(current_profile)) * time_step

    def calculate_efc(self, current_profile, time_step):
        """
        Calculate the Equivalent Full Cycles (EFC) from the current profile.
        :return: Equivalent Full Cycles.
        """
        return self.calculate_ampere_hour_throughput(current_profile, time_step) / self.cap / 2

    def calendar_aging(self, volt, temp, aging_step, start_cal):
        """
        Calculate calendar aging capacity fade.
        :param SOC: State of Charge array.
        :param temp: Temperature array in Kelvin.
        :param aging_step: Time step for aging calculation, in hours.
        :param start_cal: Initial calendar capacity.
        :return: Calendar aging capacity fade.
        """
        # Calendar aging factor
        cal_aging_factor = (
            (self.calendar_params["a"] * volt + self.calendar_params["b"])
            * np.exp(self.calendar_params["c"] / temp)
        )
        cal_aging_factor = np.maximum(cal_aging_factor, 0)  # Prevent negative aging factors

        # Cumulative calendar aging
        cal_aging_cumulative = np.sum(cal_aging_factor ** (1 / self.t_factor))

        # Calendar capacity calculation
        cal_aging = (
            (start_cal ** (1 / self.t_factor)) + aging_step/24 * cal_aging_cumulative
        ) ** self.t_factor

        return cal_aging

    def cyclic_aging(self, volt, dod, efc, start_cyc):
        """
        Calculate cyclic aging capacity fade.
        :param volt: Average voltage of this aging step.
        :param dod: DOD of this aging step.
        :param efc: EFC of this aging step.
        :param start_cyc: Initial cyclic aging of thie aging step, from 0 to 1.
        :return: Cyclic aging after this aging step, from 0 to 1.
        """
        # Cyclic aging factor
        cyc_aging_factor = (
            self.cyclic_params["a"] * (volt + self.cyclic_params["minv"]) ** 2
            + (self.cyclic_params["b"] + 0.1488)
            + self.cyclic_params["c"] * dod
        )

        # Normalize of the cyclic aging factor
        cyc_aging_factor = np.interp(cyc_aging_factor, [0, 1], [-0.01, 0])

        # Cumulative cyclic aging
        cyc_aging_cumulative = np.sum(
            cyc_aging_factor ** (1 / self.q_factor) * np.abs(efc)
        )

        # Cyclic capacity calculation
        cyc_aging = (
            (start_cyc ** (1 / self.q_factor)) + cyc_aging_cumulative
        ) ** self.q_factor

        return cyc_aging

    def create_one_aging_step(self, SOC, temp, current_profile, time_step):
        """
        Create one aging step by aggregating the stress factors.
        :param SOC: State of Charge array.
        :param temp: Temperature array in Kelvin.
        :param current_profile: Current profile array (in Amperes).
        :return: Tuple of voltage, DOD, temperature, and Ampere-hour throughput.
        """
        volt, dod = self.calculate_voltage_dod(SOC)
        temp = np.nanmean(temp)
        efc = self.calculate_efc(current_profile, time_step)
        return volt, dod, temp, efc

    def generate_current_profile(self, SOC, time_step):
        """
        Generate a current profile from the SOC array.
        :param SOC: State of Charge array.
        :param time_step: Time step for aging calculation (in hours).
        :return: Current profile array.
        """
        current_profile = (SOC[1:] - SOC[:-1]) / 100 * self.cap / time_step  # Current profile in Amperes
        return current_profile

    def count_dod(self, SOC):
        """
        Count the number of cycles and Depth of Discharge (DOD) from the SOC array.
        :param SOC: State of Charge array.
        :return: Tuple of cycle count and DOD array.
        """
        # Rainflow cycle counting
        if len(SOC) == 2:
            return abs(SOC[1] - SOC[0])
        cycles = rainflow.count_cycles(SOC)
        average_dod = np.sum([c[0] * c[1] for c in cycles]) / np.sum([c[1] for c in cycles])
        # mix the microcycles and macrocycles
        # dod_mix = 0.5 * micro_dod + 0.5 * (SOC.max() - SOC.min())
        return average_dod

    def get_capacity_fade(self, SOC, start_aging_cal, start_aging_cyc, temp=298):
        temp = np.full(len(SOC), temp)
        current_profile = self.generate_current_profile(SOC, self.time_step)
        aging_step = self.time_step * (SOC.size - 1)  # The period of one aging step, set to be equal to one charging process
        volt, dod, temp, efc = self.create_one_aging_step(SOC, temp, current_profile, self.time_step)

        # normalization of voltage and dod for cyclic aging
        dod_norm = np.interp(dod, [0, 100], [0, 1])
        volt_norm = np.interp(volt, [self.ocv_data["Voltage"].min(), self.ocv_data["Voltage"].max()], [0, 1])

        aging_cal = self.calendar_aging(volt, temp, aging_step, start_aging_cal)
        aging_cyc = self.cyclic_aging(volt_norm, dod_norm, efc, start_aging_cyc)

        return aging_cal, aging_cyc

    def total_cycle_number(self, SOC_, time_step, EOL, temp_=298):
        """
        Calculate the total cycle number until the EOL is reached.
        :param SOC_: State of Charge array.
        :param time_step: Time step for aging calculation (in hours).
        :return: Total cycle number until the EOL is reached.
        """
        SOC, driving_SOC = SOC_
        temp = np.full(len(SOC), temp_)
        temp_driving = np.full(len(driving_SOC), temp_)
        current_profile = self.generate_current_profile(SOC, time_step)
        current_profile_driving = self.generate_current_profile(driving_SOC, time_step)
        aging_step = time_step * (SOC.size - 1)  # The period of one aging step, set to be equal to one charging process
        aging_step_driving = time_step * (driving_SOC.size - 1)
        volt, dod, temp, efc = self.create_one_aging_step(SOC, temp, current_profile, time_step)
        volt_driving, dod_driving, temp_driving, efc_driving = self.create_one_aging_step(driving_SOC, temp_driving, current_profile_driving, time_step)
        # normalization of voltage and dod for cyclic aging
        dod_norm = np.interp(dod, [0, 100], [0, 1])
        volt_norm = np.interp(volt, [self.ocv_data["Voltage"].min(), self.ocv_data["Voltage"].max()], [0, 1])
        dod_norm_driving = np.interp(dod_driving, [0, 100], [0, 1])
        volt_norm_driving = np.interp(volt_driving, [self.ocv_data["Voltage"].min(), self.ocv_data["Voltage"].max()], [0, 1])

        start_aging_cal = 0
        start_aging_cyc = 0
        cycle_number = 0
        # min_aging_cost = 0.01 euro/kWh @ 60 kWh capacity and 100 euro/kWh battery cost
        # max_cycle_number = (cap * battery_cost) / min_aging_cost
        max_cycle_number = 60 * 100 / 0.01 / (efc * 2 * 60) if efc != 0 else 100000
        while True:
            cycle_number += 1
            # Calculate the charging process
            aging_cal = self.calendar_aging(volt, temp, aging_step, start_aging_cal)
            aging_cyc = self.cyclic_aging(volt_norm, dod_norm, efc, start_aging_cyc)
            # Calculate the driving process
            aging_cal = self.calendar_aging(volt_driving, temp_driving, aging_step_driving, aging_cal)
            aging_cyc = self.cyclic_aging(volt_norm_driving, dod_norm_driving, efc_driving, aging_cyc)
            start_aging_cal = aging_cal
            start_aging_cyc = aging_cyc
            if (aging_cyc + aging_cal) >= 1 - EOL:
                break
            if cycle_number > max_cycle_number:
                break
        return cycle_number, aging_cal, aging_cyc

    def calc_episode_reward(self, SOC, charging_duration):
        """
        Calculate the reward of the aging process.
        :param SOC: State of Charge array.
        :param time_step: Time step for aging calculation (in hours).
        :param battery_cost: Battery cost in Euro.
        :param charging_duration: Charging duration in number of time steps.
        :return: Reward of the aging process.
        """
        driving_SOC = np.linspace(80, SOC[0], num=int(24/self.time_step - charging_duration + 1))
        SOC_extend = np.linspace(SOC[-1], 80, num=int(charging_duration-len(SOC))+2)
        SOC = np.concatenate((SOC, SOC_extend[1:]))
        cycle_number, _, _ = self.total_cycle_number((SOC, driving_SOC), self.time_step, self.EOL)
        return -self.battery_cost / cycle_number

    def calc_immediate_reward(self, SOC, charging_duration):
        """
        Calculate the immediate reward of one step.
        :param SOC: State of Charge array.
        :param time_step: Time step for aging calculation (in hours).
        :param EOL: End of Life criteria, from 0 to 1.
        :return: Immediate reward.
        """
        current_episode_reward = self.calc_episode_reward(SOC, charging_duration)
        step_reward = current_episode_reward - self.reward_memory
        self.reward_memory = current_episode_reward
        # Calculate the cumulative current DOD
        if (SOC[-1] - SOC[-2]) * self.cum_current_dod >= 0:
            self.cum_current_dod += SOC[-1] - SOC[-2]
        else:
            self.cum_current_dod = SOC[-1] - SOC[-2]
        # Calculate the mean DOD
        self.mean_dod = self.count_dod(SOC)
        # Calculate EFC
        current_profile = self.generate_current_profile(SOC, self.time_step)
        self.current_efc = self.calculate_efc(current_profile, self.time_step)
        return step_reward

