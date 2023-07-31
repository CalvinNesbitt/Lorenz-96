"""
Model Definition for the L96 model. Based on equation 1 of Lorenz's 2005 paper (https://doi.org/10.1175/JAS3430.1)
"""
# Imports
from chaos_explorer.integrator import OdeIntegrator
from chaos_explorer.observers.xarray import TrajectoryObserver

import numpy as np
import numpy.random as rm
import xarray as xr


# Standard Parameter Choice based on LE98.
F = 8
N = 40


def l96_rhs(x, F=F, N=N):
    return np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + F


class L96Integrator(OdeIntegrator):
    def __init__(self, F=F, N=N, ic=None):
        self.F = F
        self.N = N

        # Initialise random ic if none given
        if ic is None:
            ic = rm.normal(loc=6.0, scale=2.0, size=self.N)

        super().__init__(l96_rhs, ic, parameters={"F": F, "N": N})


class L96TrajectoryObserver(TrajectoryObserver):
    def __init__(self, l96_integrator):
        super().__init__(l96_integrator)

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
