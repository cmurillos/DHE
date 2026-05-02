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
    # Predicados de pertenencia a fronteras
    # ------------------------------------------------------------------

    def on_C1(self, x, y, z):
        """C1 = {r = r1, 0 < z < L} (pared exterior del pozo, interfaz Ω1↔Ω2)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (np.abs(r - self.r1) < tol) & (z > -tol) & (z < self.L + tol)

    def on_C2(self, x, y, z):
        """C2 = {r = r0, 0 < z < L} (pared interior tubería, interfaz Ω2↔Ω3)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (np.abs(r - self.r0) < tol) & (z > -tol) & (z < self.L + tol)

    def on_C_L2(self, x, y, z):
        """C_L,2 = {r0 ≤ r ≤ r1, z = L} (fondo del anular)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (
            (r >= self.r0 - tol) & (r <= self.r1 + tol) &
            (np.abs(z - self.L) < tol)
        )

    def on_C_L3(self, x, y, z):
        """C_L,3 = {0 ≤ r ≤ r0, z = L} (fondo del tubo interior)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (r <= self.r0 + tol) & (np.abs(z - self.L) < tol)

    def on_C_L(self, x, y, z):
        """C_L = C_L,2 ∪ C_L,3."""
        return self.on_C_L2(x, y, z) | self.on_C_L3(x, y, z)

    def on_D1(self, x, y, z):
        """D1 = {0 ≤ r ≤ R, z = h} (frontera geotérmica profunda)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return np.abs(z - self.h) < self.tol

    def on_D2_omega1(self, x, y, z):
        """D2 ∩ Ω̄1 = {r1 ≤ r ≤ R, z = 0} (superficie páramo / contacto suelo-atmósfera)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (
            (r >= self.r1 - tol) & (r <= self.R + tol) &
            (np.abs(z) < tol)
        )

    def on_D2_omega2(self, x, y, z):
        """D2 ∩ Ω̄2 = {r0 ≤ r ≤ r1, z = 0} (entrada del fluido al anular)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (
            (r >= self.r0 - tol) & (r <= self.r1 + tol) &
            (np.abs(z) < tol)
        )

    def on_D2_omega3(self, x, y, z):
        """D2 ∩ Ω̄3 = {0 ≤ r ≤ r0, z = 0} (salida del fluido del tubo interior)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        return (r <= self.r0 + tol) & (np.abs(z) < tol)

    def on_Gamma_ext(self, x, y, z):
        """Γ_ext = {r = R, 0 < z < h} (frontera lateral exterior, flujo nulo)."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x**2 + y**2)
        tol = self.tol
        # z=0 es D2 y z=h es D1; Γ_ext es estrictamente entre ambos
        return (np.abs(r - self.R) < tol) & (z > tol) & (z < self.h - tol)

    # ------------------------------------------------------------------
    # Helpers de etiquetado para mallas
    # ------------------------------------------------------------------

    def label_cells(self, cell_centroids):
        """
        Dado array (N_cells, 3) con centroides, retorna array entero (N_cells,):
            1 -> Ω1,  2 -> Ω2,  3 -> Ω3,  0 -> sin clasificar (warning).
        """
        cell_centroids = np.asarray(cell_centroids)
        x, y, z = cell_centroids[:, 0], cell_centroids[:, 1], cell_centroids[:, 2]
        labels = np.zeros(len(x), dtype=int)

        # Orden importa: Ω2 y Ω3 son más restrictivos que Ω1 en z<L
        m3 = self.in_omega3(x, y, z)
        m2 = self.in_omega2(x, y, z) & ~m3
        m1 = self.in_omega1(x, y, z) & ~m2 & ~m3

        labels[m1] = 1
        labels[m2] = 2
        labels[m3] = 3

        n_unclassified = np.sum(labels == 0)
        if n_unclassified > 0:
            warnings.warn(
                f"{n_unclassified} celdas sin clasificar (tag=0). "
                "Verificar la malla y los predicados geométricos.",
                stacklevel=2,
            )
        return labels

    def label_facets(self, facet_centroids):
        """
        Dado array (N_facets, 3) con centroides de caras de frontera,
        retorna array entero (N_facets,):
            10 -> C1,  20 -> C2,  31 -> C_L2,  32 -> C_L3,
            41 -> D2∩Ω̄1,  42 -> D2∩Ω̄2,  43 -> D2∩Ω̄3,
            50 -> D1,  60 -> Γ_ext,  0 -> sin clasificar.
        """
        facet_centroids = np.asarray(facet_centroids)
        x, y, z = facet_centroids[:, 0], facet_centroids[:, 1], facet_centroids[:, 2]
        labels = np.zeros(len(x), dtype=int)

        labels[self.on_C1(x, y, z)] = 10
        labels[self.on_C2(x, y, z)] = 20
        labels[self.on_C_L2(x, y, z)] = 31
        labels[self.on_C_L3(x, y, z)] = 32
        labels[self.on_D2_omega1(x, y, z)] = 41
        labels[self.on_D2_omega2(x, y, z)] = 42
        labels[self.on_D2_omega3(x, y, z)] = 43
        labels[self.on_D1(x, y, z)] = 50
        labels[self.on_Gamma_ext(x, y, z)] = 60

        return labels
