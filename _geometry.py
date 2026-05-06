"""
_geometry.py
============

Define los subdominios Ω1, Ω2, Ω3 y las fronteras C1, C2, C_L,
D1, D2, Γ_ext del modelo acoplado, así como predicados booleanos
sobre puntos (x, y, z) que dicen a qué conjunto pertenece cada punto.

Toda la información geométrica del modelo (radios, profundidades,
tolerancias) vive aquí.
"""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DHEGeometry:
    """
    Parámetros geométricos del modelo DHE acoplado.

    Convención de coordenadas
    -------------------------
    z = 0 es la superficie del páramo (parte de arriba del dominio).
    z = h es la frontera profunda inferior.
    z crece HACIA ABAJO.

    Parameters
    ----------
    r0 : float
        Radio interior de la tubería coaxial (radio de Ω3). 0 < r0 < r1.
    r1 : float
        Radio exterior de la tubería coaxial (interfaz suelo-tubería). r0 < r1 < R.
    R : float
        Radio exterior del dominio geológico modelado.
    L : float
        Profundidad activa del intercambiador. 0 < L < h.
    h : float
        Profundidad total del dominio geológico modelado.
    tol : float
        Tolerancia geométrica para los predicados booleanos (default: 1e-6).
    """
    r0: float
    r1: float
    R: float
    L: float
    h: float
    tol: float = 1e-6

    def __post_init__(self):
        if not (0 < self.r0):
            raise ValueError(f"r0 debe ser > 0, got {self.r0}")
        if not (self.r0 < self.r1):
            raise ValueError(f"r0 < r1 requerido, got r0={self.r0}, r1={self.r1}")
        if not (self.r1 < self.R):
            raise ValueError(f"r1 < R requerido, got r1={self.r1}, R={self.R}")
        if not (0 < self.L):
            raise ValueError(f"L debe ser > 0, got {self.L}")
        if not (self.L < self.h):
            raise ValueError(f"L < h requerido, got L={self.L}, h={self.h}")

    # ------------------------------------------------------------------
    # Predicados de pertenencia a subdominios
    # ------------------------------------------------------------------

    def in_omega1(self, x, y, z):
        """
        Ω1 = {r1 < r < R, 0 < z < h} ∪ {0 ≤ r < r1, L < z < h}

        La unión: arriba (z<L) sólo la corona externa; abajo (z>L) TODO el
        cilindro es subsuelo (incluyendo el centro).
        """
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        in_z = (z > -tol) & (z < self.h + tol)
        corona = (r > self.r1 - tol) & (r < self.R + tol) & in_z
        tapon = (r < self.r1 + tol) & (z > self.L - tol) & (z < self.h + tol)
        return corona | tapon

    def in_omega2(self, x, y, z):
        """Ω2 = {r0 < r < r1, 0 < z < L} (anular descendente)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (
            (r > self.r0 - tol) & (r < self.r1 + tol) &
            (z > -tol) & (z < self.L + tol)
        )

    def in_omega3(self, x, y, z):
        """Ω3 = {0 ≤ r < r0, 0 < z < L} (tubo interior ascendente)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (r < self.r0 + tol) & (z > -tol) & (z < self.L + tol)

    # ------------------------------------------------------------------
