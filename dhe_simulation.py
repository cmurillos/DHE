"""
DHE_simulation
==============

Clase principal que orquesta la simulación térmica acoplada del DHE.
"""

import numpy as np
import pandas as pd

from ._geometry import DHEGeometry
from ._coupled_mesh import CoupledCylinderMesh
from ._coupled_fem_solver import CoupledCylinderFEMSolver
from ._generador_campos import GeneradorCamposFisicos
from ._surface_temperature import SurfaceTemperature
from ._result import DHEResult


class DHE_simulation:
    """
    Simulación térmica acoplada de un Downhole Heat Exchanger.

    Parameters
    ----------
    csv : str
        Ruta al CSV con propiedades del subsuelo.
    r0, r1, R : float
        Radios geométricos: 0 < r0 < r1 < R.
    L, h : float
        Profundidades: 0 < L < h.
    alpha2, alpha3 : float
        Difusividades térmicas de los fluidos [m²/s].
    U_z : float
        Velocidad axial del fluido [m/s].
    delta_0, delta_1, delta_2, delta_f : float
        Coeficientes Robin en D2, C1, C2, D1.
    T_int : float
        Temperatura del fluido inyectado [K].
    T_f : float
        Temperatura geotérmica profunda [K].
    Ts : SurfaceTemperature or callable or float
        Temperatura superficial del páramo.
    nr_omega1, nr_omega2, nr_omega3 : int
        Resolución radial por subdominio.
    nz_above_L, nz_below_L : int
        Resolución axial.
    ntheta : int
        Divisiones angulares.
    tries : int
        Número de realizaciones Monte Carlo del campo G.
    seed : int or None
        Semilla para reproducibilidad.
    """

    def __init__(
        self,
        csv,
        r0, r1, R,
        L, h,
        alpha2, alpha3, U_z,
        delta_0, delta_1, delta_2, delta_f,
        T_int, T_f,
        Ts,
        nr_omega1=8, nr_omega2=4, nr_omega3=4,
        nz_above_L=20, nz_below_L=10, ntheta=16,
        tries=1, seed=None,
    ):
        self.csv = csv
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.U_z = U_z
        self.delta_0 = delta_0
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.delta_f = delta_f
        self.T_int = T_int
        self.T_f = T_f
        self.tries = tries
        self.seed = seed

        # Validar CSV
        df = pd.read_csv(csv)
        required = {'Z', 'T', 'Tvar', 'rho', 'rhovar', 'c', 'cvar', 'k', 'kvar'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV falta columnas: {missing}")

        # Normalizar Ts
        if isinstance(Ts, (int, float)):
            self.Ts = SurfaceTemperature.constant(float(Ts))
        elif callable(Ts) and not isinstance(Ts, SurfaceTemperature):
            self.Ts = SurfaceTemperature.from_callable(Ts)
        else:
            self.Ts = Ts

        # Geometría
        self.geometry = DHEGeometry(r0=r0, r1=r1, R=R, L=L, h=h)

        # Malla (una sola, reutilizable entre realizaciones)
        print("Construyendo malla...", flush=True)
