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
        self.mesh_obj = CoupledCylinderMesh(
            self.geometry,
            nr_omega1=nr_omega1,
            nr_omega2=nr_omega2,
            nr_omega3=nr_omega3,
            nz_above_L=nz_above_L,
            nz_below_L=nz_below_L,
            ntheta=ntheta,
        )
        print(f"  Nodos: {self.mesh_obj.nodes.shape[0]} | "
              f"Tetraedros: {self.mesh_obj.elements.shape[0]}", flush=True)

        # Solver (reutilizable; sólo el ensamblaje cambia con cada realización)
        self.solver = CoupledCylinderFEMSolver(self.mesh_obj, self.geometry)

    def _generate_fields(self, seed=None):
        """Crea una realización de los campos ρ, c, k, T0."""
        rng = np.random.default_rng(seed)
        gen = GeneradorCamposFisicos(
            csv_path=self.csv,
            nodes=self.mesh_obj.nodes,
            rng=rng,
            T_f=self.T_f,
        )
        return gen.get_all()

    def run(
        self,
        dt, t_save, tf,
        t_on=0.0, t_off=None,
        stabilize_first=True,
        output_prefix='simulation',
    ):
        """
        Ejecuta `tries` simulaciones y retorna lista de DHEResult
        (o un solo DHEResult si tries==1).

        Parameters
        ----------
        dt : float — paso temporal [s]
        t_save : float — intervalo de guardado [s]
        tf : float — tiempo final [s]
        t_on, t_off : float — ventana de operación del DHE
        stabilize_first : bool
        output_prefix : str — prefijo para archivos .npz

        Returns
        -------
        DHEResult or list[DHEResult]
        """
        results = []

        for trial in range(self.tries):
            seed_i = None if self.seed is None else self.seed + trial
            print(f"\n=== Realización {trial+1}/{self.tries} (seed={seed_i}) ===",
                  flush=True)

            rho_f, c_f, k_f, T0_f, T_f_inferred = self._generate_fields(seed_i)
            T_f = self.T_f if self.T_f is not None else T_f_inferred

            print("Ensamblando sistema FEM...", flush=True)
            self.solver.assemble_system(
                rho_func=rho_f,
                c_func=c_f,
                k_func=k_f,
                alpha2=self.alpha2,
                alpha3=self.alpha3,
                U_z=self.U_z,
                delta_0=self.delta_0,
                delta_1=self.delta_1,
                delta_2=self.delta_2,
                delta_f=self.delta_f,
                T_f=T_f,
            )

            result = self.solver.solve(
                T0_func=T0_f,
                dt=dt,
                tf=tf,
                t_save=t_save,
                Ts_func=self.Ts,
                T_int=self.T_int,
                t_on=t_on,
                t_off=t_off,
                stabilize_first=stabilize_first,
            )

            fname = f"{output_prefix}_trial{trial+1:03d}"
            result.save(fname)
            print(f"  Guardado: {fname}.npz", flush=True)

            results.append(result)

        if self.tries == 1:
            return results[0]
        return results
