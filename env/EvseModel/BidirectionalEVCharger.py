import json
import numpy as np


import json
import numpy as np


class BidirectionalEVCharger:
    def __init__(self, config_file, rated_power):
        """
        Initialize the charger with parameters from a JSON config file.
        :param config_file: Path to the JSON file containing technical values
        :param rated_power: Rated power of the charger (kW)
        """
        self.P_grid_max = rated_power  # Rated power of the charger (kW)
        with open(config_file, 'r') as file:
            config = json.load(file)

        # Value from https://doi.org/10.1016/j.segy.2024.100145
        self.rated_power_model_ac_side = config.get("rated_power")
        self.loss_parameters = config.get("loss_parameters")

        self.P_dc_cha_max, self.P_loss_cha_max, self.P_dc_dch_max, self.P_loss_dch_max = self.get_max_value()
        self.P_ac_max_ori, self.P_dc_cha_max_ori, self.P_loss_cha_max_ori, self.P_dc_dch_max_ori, \
        self.P_loss_dch_max_ori = self.get_max_value_original_model()

        self.param_cha_dc_to_ac = None
        self.param_dch_dc_to_ac = None
        self.eval_normalized_loss_parameter()

    def get_max_value(self):
        """
        Calculate the maximum power of EV and the maximum power loss.
        """
        # charging process
        # P_grid_max = P_EV_max + P_loss_max
        # P_grid_max = 0.003 * P_EV_max**2 + (0.093 + 1) * P_EV_max + 0.05
        P_ev_max_cha = np.roots([self.loss_parameters['charging'][0], self.loss_parameters['charging'][1]+1, self.loss_parameters['charging'][2]-self.P_grid_max])
        try:
            P_ev_max_cha = min(filter(lambda x: x.imag == 0 and x.real >= 0, P_ev_max_cha),
                       key=lambda x: abs(x - self.P_grid_max)).real
        except ValueError:
            raise ValueError('P_ev_max can not be determined...')
        P_loss_max_cha = self.loss_parameters['charging'][0] * P_ev_max_cha**2 + self.loss_parameters['charging'][1] * P_ev_max_cha + self.loss_parameters['charging'][2]

        # discharging process
        # P_grid_max = P_EV_max - P_loss_max
        # P_grid_max = -0.005 * P_EV_max**2 + (1 - 0.063) * P_EV_max - 0.036
        P_ev_max_dch = np.roots([-self.loss_parameters['discharging'][0], 1 - self.loss_parameters['discharging'][1], -self.loss_parameters['discharging'][2]-self.P_grid_max])
        try:
            P_ev_max_dch = min(filter(lambda x: x.imag == 0 and x.real >= 0, P_ev_max_dch),
                       key=lambda x: abs(x - self.P_grid_max)).real
        except ValueError:
            raise ValueError('P_ev_max can not be determined...')
        P_loss_max_dch = self.loss_parameters['discharging'][0] * P_ev_max_dch**2 + self.loss_parameters['discharging'][1] * P_ev_max_dch + self.loss_parameters['discharging'][2]

        return P_ev_max_cha, P_loss_max_cha, P_ev_max_dch, P_loss_max_dch

    def get_max_value_original_model(self):
        # charge
        P_ac_max_ori = self.rated_power_model_ac_side
        P_dc_cha_max_ori = np.roots([self.loss_parameters['charging'][0], self.loss_parameters['charging'][1]+1, self.loss_parameters['charging'][2]-P_ac_max_ori])
        try:
            P_dc_cha_max_ori = min(filter(lambda x: x.imag == 0 and x.real >= 0, P_dc_cha_max_ori),
                       key=lambda x: abs(x - P_ac_max_ori)).real
        except ValueError:
            raise ValueError('P_dc_cha_max_ori can not be determined...')
        P_loss_cha_max_ori = P_ac_max_ori - P_dc_cha_max_ori
        assert P_loss_cha_max_ori >= 0, "P_loss_cha_max_ori should be greater than 0"

        # discharge
        P_dc_dch_max_ori = np.roots([-self.loss_parameters['discharging'][0], 1 - self.loss_parameters['discharging'][1], -self.loss_parameters['discharging'][2]-P_ac_max_ori])
        try:
            P_dc_dch_max_ori = min(filter(lambda x: x.imag == 0 and x.real >= 0, P_dc_dch_max_ori),
                       key=lambda x: abs(x - P_ac_max_ori)).real
        except ValueError:
            raise ValueError('P_dc_dch_max_ori can not be determined...')
        P_loss_dch_max_ori = P_dc_dch_max_ori - P_ac_max_ori
        assert P_loss_dch_max_ori >= 0, "P_loss_dch_max_ori should be greater than 0"

        return P_ac_max_ori, P_dc_cha_max_ori, P_loss_cha_max_ori, P_dc_dch_max_ori, P_loss_dch_max_ori

    def eval_normalized_loss_parameter(self):
        # charge
        # dc to ac
        denominator = self.loss_parameters["charging"][0] * self.P_dc_cha_max_ori**2 + \
                      self.loss_parameters["charging"][1] * self.P_dc_cha_max_ori + \
                      self.loss_parameters["charging"][2]
        param_cha_dc_to_loss = [self.loss_parameters["charging"][0] * self.P_dc_cha_max_ori**2 / denominator,
                                self.loss_parameters["charging"][1] * self.P_dc_cha_max_ori / denominator,
                                self.loss_parameters["charging"][2] / denominator]
        self.param_cha_dc_to_ac = [param_cha_dc_to_loss[0] * self.P_loss_cha_max / self.P_dc_cha_max**2,
                                   param_cha_dc_to_loss[1] * self.P_loss_cha_max / self.P_dc_cha_max + 1,
                                   param_cha_dc_to_loss[2] * self.P_loss_cha_max]
        # ac to dc is the reverse of dc to ac

        # discharge
        # dc to ac
        denominator = self.loss_parameters["discharging"][0] * self.P_dc_dch_max_ori ** 2 + \
                      self.loss_parameters["discharging"][1] * self.P_dc_dch_max_ori + \
                      self.loss_parameters["discharging"][2]
        param_dch_dc_to_loss = [self.loss_parameters["discharging"][0] * self.P_dc_dch_max_ori ** 2 / denominator,
                                self.loss_parameters["discharging"][1] * self.P_dc_dch_max_ori / denominator,
                                self.loss_parameters["discharging"][2] / denominator]
        self.param_dch_dc_to_ac = [-param_dch_dc_to_loss[0] * self.P_loss_dch_max / self.P_dc_dch_max**2,
                                   1 - param_dch_dc_to_loss[1] * self.P_loss_dch_max / self.P_dc_dch_max,
                                   -param_dch_dc_to_loss[2] * self.P_loss_dch_max]
        # ac to dc is the reverse of dc to ac

    def get_power_ev_side(self, p_ac):
        """
        Calculate the power on the EV side if the grid side power is known.
        """
        if p_ac > 0:
            # charge process: use the reverse function of self.param_cha_dc_to_ac
            p_dc = np.roots([self.param_cha_dc_to_ac[0], self.param_cha_dc_to_ac[1], self.param_cha_dc_to_ac[2] - p_ac])
            try:
                p_dc = min(filter(lambda x: x.imag == 0 and x.real >= 0, p_dc), key=lambda x: abs(x - p_ac)).real
            except ValueError:
                p_dc = 0
        elif p_ac < 0:
            # discharge process: use the reverse function of self.param_dch_dc_to_ac
            p_ac = -p_ac
            p_dc = np.roots([self.param_dch_dc_to_ac[0], self.param_dch_dc_to_ac[1], self.param_dch_dc_to_ac[2] - p_ac])
            try:
                p_dc = min(filter(lambda x: x.imag == 0 and x.real >= 0, p_dc), key=lambda x: abs(x - p_ac)).real
            except ValueError:
                p_dc = 0
            p_dc = -p_dc
        else:
            p_dc = 0
        return p_dc

    def get_power_grid_side(self, p_dc):
        """
        Calculate the power on the grid side if the EV side power is known.
        """
        if p_dc > 0:
            # charge process: use the function self.param_cha_dc_to_ac
            p_ac = self.param_cha_dc_to_ac[0] * p_dc**2 + self.param_cha_dc_to_ac[1] * p_dc + self.param_cha_dc_to_ac[2]
        elif p_dc < 0:
            # discharge process: use the function self.param_dch_dc_to_ac
            p_dc = -p_dc
            p_ac = self.param_dch_dc_to_ac[0] * p_dc**2 + self.param_dch_dc_to_ac[1] * p_dc + self.param_dch_dc_to_ac[2]
            p_ac = -p_ac
        else:
            p_ac = 0
        if p_dc * p_ac < 0:
            p_ac = 0
        return p_ac

    def get_efficiency(self, p_ac):
        """
        Calculate the efficiency of the conversion process.
        """
        p_dc = self.get_power_ev_side(p_ac)
        if p_ac > 0:
            efficiency = p_dc / p_ac
        elif p_ac < 0:
            efficiency = p_ac / p_dc
        else:
            efficiency = 0
        return efficiency

    def get_power_loss(self, p_ac):
        """
        Calculate power loss based on the mode (charging or discharging).
        """
        return p_ac - self.get_power_dc_side(p_ac)




