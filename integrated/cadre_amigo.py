"""AMIGO components shared by the integrated CADRE model."""

import amigo as am
import numpy as np


class ReactionWheelInitialCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("w_RW", shape=(3,))

        self.add_constraint("res", shape=(3,))

    def compute(self):
        w_RW = self.inputs["w_RW"]

        self.constraints["res"] = [w_RW[0], w_RW[1], w_RW[2]]


class BatteryDynamicsLinked(am.Component):
    """
    CADRE battery dynamics with P_bat and temperature
    supplied by the coupled AMIGO model.
    """

    def __init__(self):
        super().__init__()

        self.add_constant("sigma", value=1.0e-10)
        self.add_constant("eta", value=0.99)
        self.add_constant("Cp", value=2900.0 * 0.001 * 3600.0)
        self.add_constant("IR", value=0.9)
        self.add_constant("T0", value=293.0)
        self.add_constant("alpha", value=np.log(1.0 / 1.1**5))
        self.add_constant("e_minus_1", value=np.e - 1.0)

        self.add_input("P_bat")
        self.add_input("temperature")
        self.add_input("SOC")
        self.add_input("SOCdot")
        self.add_input("I_bat")

        self.add_constraint("res")
        self.add_constraint("current_res")

    def compute(self):
        sigma = self.constants["sigma"]
        eta = self.constants["eta"]
        Cp = self.constants["Cp"]
        IR = self.constants["IR"]
        T0 = self.constants["T0"]
        alpha = self.constants["alpha"]
        e_minus_1 = self.constants["e_minus_1"]

        P_bat = self.inputs["P_bat"]
        temperature = self.inputs["temperature"]

        SOC = self.inputs["SOC"]
        SOCdot = self.inputs["SOCdot"]

        voc = 3.0 + (am.exp(SOC) - 1.0) / e_minus_1

        voltage = IR * voc * (2.0 - am.exp(alpha * (temperature - T0) / T0))

        current = P_bat / voltage
        self.constraints["current_res"] = current - self.inputs["I_bat"]

        rate = -sigma / 24.0 * SOC + eta / Cp * current
        self.constraints["res"] = rate - SOCdot


class BatteryInitialCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("initial_soc")
        self.add_input("SOC")

        self.add_constraint("res")

    def compute(self):
        SOC = self.inputs["SOC"]

        self.constraints["res"] = SOC - self.inputs["initial_soc"]


class BatteryPeriodicity(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("SOC_initial")
        self.add_input("SOC_final")

        self.add_constraint("res")

    def compute(self):
        self.constraints["res"] = self.inputs["SOC_final"] - self.inputs["SOC_initial"]


class DataInitialCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("Data")

        self.add_constraint("res")

    def compute(self):
        self.constraints["res"] = self.inputs["Data"]


class DownloadObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("Data")

        self.add_objective("obj")

    def compute(self):
        # AMIGO minimizes.
        # Therefore minimize -Data to maximize downloaded data.
        self.objective["obj"] = -self.inputs["Data"]
