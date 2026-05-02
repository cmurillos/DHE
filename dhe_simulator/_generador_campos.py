"""
_generador_campos.py
====================

Construye campos heterogéneos ρ(x,y,z), c(x,y,z), k(x,y,z), T0(x,y,z)
desde un CSV con perfiles estratificados (Z, T, Tvar, rho, rhovar,
c, cvar, k, kvar).

Para cada campo f ∈ {ρ, c, k, T0}:
    1. f0(z): constante a trozos interpolada desde la tabla.
    2. G(x,y,z): campo aleatorio con media ~0 y varianza ~1 por nivel z,
       construido muestreando gaussiana en cada nodo y usando cKDTree.
    3. f(x,y,z) = f0(z) + var_f(z) * G(x,y,z).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class GeneradorCamposFisicos:
    """
    Lee un CSV estratigráfico y construye los campos heterogéneos
    del subsuelo como funciones evaluables en (x, y, z).

    Parameters
    ----------
    csv_path : str
        Ruta al CSV con columnas: Z, T, Tvar, rho, rhovar, c, cvar, k, kvar.
    nodes : ndarray (N, 3)
        Coordenadas (x, y, z) de los nodos de la malla. Se usan para
        construir el campo aleatorio G.
    rng : np.random.Generator or None
        Generador de números aleatorios. Si None se usa np.random.default_rng().
    T_f : float or None
        Temperatura geotérmica de fondo. Si None se infiere como T0(z=h).
    """

    def __init__(self, csv_path, nodes, rng=None, T_f=None):
        self.nodes = np.asarray(nodes)
        self.rng = rng if rng is not None else np.random.default_rng()

        df = pd.read_csv(csv_path)
        required = {'Z', 'T', 'Tvar', 'rho', 'rhovar', 'c', 'cvar', 'k', 'kvar'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV falta columnas: {missing}")

        df = df.sort_values('Z').reset_index(drop=True)
        self.df = df

        # Profundidades de los estratos
        self.z_strata = df['Z'].values

        # Construir el campo aleatorio G (una realización por nodo)
        self._build_random_field()

        # Construir callables para cada campo físico
        self.rho_func = self._make_field('rho', 'rhovar')
        self.c_func = self._make_field('c', 'cvar')
        self.k_func = self._make_field('k', 'kvar')
        self.T0_func = self._make_field('T', 'Tvar')

        # Temperatura de fondo
        if T_f is not None:
            self.T_f = T_f
        else:
            # Inferir como la temperatura del estrato más profundo
            self.T_f = float(df['T'].iloc[-1])

    def _piecewise_constant(self, z_vals, column):
        """
        Interpola columna del CSV de forma constante a trozos:
        para z ∈ [z_i, z_{i+1}) retorna df[column][i].
        Para z < z_strata[0] retorna df[column][0].
        Para z >= z_strata[-1] retorna df[column][-1].
        """
        z_vals = np.asarray(z_vals)
        values = self.df[column].values
        # np.searchsorted con 'right': idx = número de estratos con Z <= z
        idx = np.searchsorted(self.z_strata, z_vals, side='right') - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    def _build_random_field(self):
        """
        Construye G en cada nodo: muestrea N(0,1) y guarda un cKDTree
        para lookup de valor más cercano.
        El campo es gaussiano con media 0 y varianza ~1 en cada nivel z
        (por construcción marginal: cada nodo tiene valor iid N(0,1)).
        """
        N = len(self.nodes)
        self._G_values = self.rng.standard_normal(N)
        self._G_tree = cKDTree(self.nodes)

    def _eval_G(self, x, y, z):
        """Evalúa G en puntos (x,y,z) usando el vecino más cercano."""
        pts = np.column_stack([x, y, z])
        _, idx = self._G_tree.query(pts)
        return self._G_values[idx]

    def _make_field(self, mean_col, var_col):
        """
        Retorna callable f(x, y, z) -> ndarray para el campo físico
        correspondiente a mean_col ± var_col * G.
        """
        def field(x, y, z):
            x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
            f0 = self._piecewise_constant(z, mean_col)
            var = self._piecewise_constant(z, var_col)
            G = self._eval_G(x, y, z)
            return f0 + var * G

        return field

    def get_all(self):
        """
        Retorna (rho_func, c_func, k_func, T0_func, T_f).
        Cada función tiene firma f(x, y, z) -> ndarray.
        """
        return self.rho_func, self.c_func, self.k_func, self.T0_func, self.T_f
