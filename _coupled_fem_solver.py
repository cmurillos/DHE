"""
_coupled_fem_solver.py
======================

Solver FEM para el sistema acoplado de 3 PDEs en Ω1 ∪ Ω2 ∪ Ω3.

Implementación con SUBMALLAS independientes por subdominio.
Cada subdominio tiene su propio MeshTet, Basis y numeración de DOFs local.
El vector de solución global es T = [T1; T2; T3] con DOFs DISTINTOS.

Las matrices de acoplamiento cruzado (B12, B21, B23, B32) se ensamblan
manualmente sobre las caras de interfaz (C1, C2) usando KD-trees para
mapear los nodos físicos a los DOF locales de cada submalla.

Sistema (Euler implícito):
    (M + dt*A) T^{n+1} = M T^n + dt * L(t^{n+1})

Dirichlet:
    - T2 = T_int en D2∩Ω̄2 (entrada del fluido)
    - T2 = T1 en C_L,2   ) eliminación de DOFs slave:
    - T3 = T1 en C_L,3   ) se sustituyen T2[CL2]→T1[CL2] y T3[CL3]→T1[CL3]
"""

import time
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from skfem import (
    ElementTetP1,
    Basis,
    FacetBasis,
    BilinearForm,
    LinearForm,
    asm,
    MeshTet,
)
from skfem.models import laplace, mass

from ._result import DHEResult


# ---------------------------------------------------------------------------
# Formas bilineales y lineales
# ---------------------------------------------------------------------------

@BilinearForm
def diffusion_k(u, v, w):
    return w.k * (u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1] + u.grad[2]*v.grad[2])


@BilinearForm
def mass_rho_c(u, v, w):
    return w.rho_c * u * v


@BilinearForm
def advection_pos_z(u, v, w):
    """∫ U_z (∂_z u) v — advección descendente."""
    return w.U_z * u.grad[2] * v


@BilinearForm
def advection_neg_z(u, v, w):
    """∫ U_z (∂_z u) v — advección ascendente (signo opuesto)."""
    return -w.U_z * u.grad[2] * v


@BilinearForm
def robin_surface(u, v, w):
    return w.delta * u * v


@LinearForm
def rhs_surface(v, w):
    return w.delta * w.T_bc * v


# ---------------------------------------------------------------------------
# Predicado con tolerancia ampliada
# ---------------------------------------------------------------------------

def _make_loose_predicate(pred_fn, factor=50):
    """
    Envuelve un predicado geométrico inyectando una tolerancia más holgada.
    Dado que los predicados del DHEGeometry usan self.tol internamente,
    aquí escalamos la comparación aproximando con un factor de escala.
    """
    # Extraer el objeto DHEGeometry del closure del predicado
    # (los predicados son métodos bound de DHEGeometry)
    import functools

    @functools.wraps(pred_fn)
    def loose(x, y, z):
        # Llamar el predicado original y OR con una versión con tol mayor
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        geo = pred_fn.__self__  # método bound
