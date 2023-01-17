"""
Model Definition for the L96 model. Based on equation 1 of Lorenz's 2005 paper (https://doi.org/10.1175/JAS3430.1)
"""

# Imports
from detDyn.dynamics.integrator import odeIntegrator
from detDyn.dynamics.observers import baseObserver

import numpy as np
import numpy.random as rm
import xarray as xr
from tqdm import tqdm

# Standard Parameter Choice based on page 1577 in Lorenz 2005.
F = 10
N = 30


def l96_rhs(x, F=10, N=30):
    return -np.roll(x, -1) * (np.roll(x, -2) - np.roll(x, 1)) - x + F


class l96Integrator(odeIntegrator):
    def __init__(self, F=10, N=30, ic=None):

        self.F = F
        self.N = N

        # Initialise random ic if none given
        if ic is None:
            ic = rm.normal(loc=6.0, scale=2.0, size=self.N)

        super().__init__(l96_rhs, ic, parameters={"F": F, "N": N})


class l96TrajectoryObserver(baseObserver):
    def __init__(self, l96_integrator):

        super().__init__(l96_integrator)

    def look(self, integrator):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self._time_obs.append(integrator.time)

        # Making Observations
        self._observations.append(integrator.state.copy())
        return

    def make_observations(self, number, frequency, timer=True):
        self.look(self.integrator)  # Make a note of IC
        for x in tqdm(range(number), disable=not timer):
            self.integrator.run(frequency)
            self.look(self.integrator)
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if len(self._observations) == 0:
            print("I have no observations! :(")
            return

        dic = {}
        _time = self._time_obs
        dic["X"] = xr.DataArray(
            self._observations,
            dims=["time", "space"],
            name="X",
            coords={"time": _time, "space": np.arange(1, 1 + self.parameters["N"])},
        )
        return xr.Dataset(dic, attrs=self.parameters)
