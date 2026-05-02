"""
Tests unitarios para DHEGeometry.
"""

import numpy as np
import pytest
from dhe_simulator._geometry import DHEGeometry


@pytest.fixture
def geo():
    return DHEGeometry(r0=0.05, r1=0.10, R=20.0, L=150.0, h=200.0)


# ---------------------------------------------------------------------------
# Tests de validación
# ---------------------------------------------------------------------------

def test_invalid_r0_r1():
    with pytest.raises(ValueError):
        DHEGeometry(r0=0.15, r1=0.10, R=20.0, L=150.0, h=200.0)


def test_invalid_L_h():
    with pytest.raises(ValueError):
        DHEGeometry(r0=0.05, r1=0.10, R=20.0, L=210.0, h=200.0)


# ---------------------------------------------------------------------------
# Tests de subdominios
# ---------------------------------------------------------------------------

def test_omega1_corona_external(geo):
    """Un punto en la corona exterior (r > r1, z < L) debe estar en Ω1."""
    x, y, z = 5.0, 0.0, 50.0  # r=5 > r1=0.10, z=50 < L=150
    assert geo.in_omega1(x, y, z)
    assert not geo.in_omega2(x, y, z)
    assert not geo.in_omega3(x, y, z)


def test_omega1_tapon_inferior_centro(geo):
    """El punto (0, 0, (L+h)/2) debe estar en Ω1 (tapón inferior, centro)."""
    z = (geo.L + geo.h) / 2  # 175.0 > L=150
    assert geo.in_omega1(0.0, 0.0, z)
    assert not geo.in_omega2(0.0, 0.0, z)
    assert not geo.in_omega3(0.0, 0.0, z)


def test_omega1_NOT_at_center_above_L(geo):
    """Un punto en (0, 0, L/2) está en el eje ARRIBA de L: pertenece a Ω3, NO a Ω1."""
    z = geo.L / 2  # 75 < L=150
    assert not geo.in_omega1(0.0, 0.0, z)
    assert geo.in_omega3(0.0, 0.0, z)


def test_omega2_annular(geo):
    """Un punto en el anular (r0 < r < r1, z < L) debe estar en Ω2."""
    r = (geo.r0 + geo.r1) / 2
    x, y, z = r, 0.0, geo.L / 2
    assert geo.in_omega2(x, y, z)
    assert not geo.in_omega1(x, y, z)
    assert not geo.in_omega3(x, y, z)


def test_omega3_inner(geo):
    """Un punto en r < r0, z < L debe estar en Ω3."""
    x, y, z = geo.r0 / 2, 0.0, geo.L / 2
    assert geo.in_omega3(x, y, z)
    assert not geo.in_omega1(x, y, z)
    assert not geo.in_omega2(x, y, z)


def test_omega2_NOT_below_L(geo):
    """Debajo de L el anular no existe — todo es Ω1."""
    r = (geo.r0 + geo.r1) / 2
    z = (geo.L + geo.h) / 2  # > L
    assert geo.in_omega1(r, 0.0, z)
    assert not geo.in_omega2(r, 0.0, z)


# ---------------------------------------------------------------------------
# Tests de fronteras
# ---------------------------------------------------------------------------

def test_C1_interface(geo):
    """Un punto en r=r1, z=L/2 debe estar en C1."""
    x, y, z = geo.r1, 0.0, geo.L / 2
    assert geo.on_C1(x, y, z)
    assert not geo.on_C2(x, y, z)


def test_C2_interface(geo):
    """Un punto en r=r0, z=L/2 debe estar en C2."""
    x, y, z = geo.r0, 0.0, geo.L / 2
    assert geo.on_C2(x, y, z)
    assert not geo.on_C1(x, y, z)


def test_C_L_bottom(geo):
    """Puntos en z=L con r ∈ [0, r1] deben estar en C_L."""
    # C_L3: r < r0
    assert geo.on_C_L3(0.02, 0.0, geo.L)
    assert geo.on_C_L(0.02, 0.0, geo.L)
    # C_L2: r0 <= r <= r1
    r_mid = (geo.r0 + geo.r1) / 2
    assert geo.on_C_L2(r_mid, 0.0, geo.L)
    assert geo.on_C_L(r_mid, 0.0, geo.L)


def test_D1_deep(geo):
    """Un punto en z=h debe estar en D1."""
    assert geo.on_D1(5.0, 0.0, geo.h)
    assert not geo.on_D1(5.0, 0.0, geo.h / 2)


def test_D2_omega1_surface(geo):
    """r1 < r < R, z=0 debe estar en D2∩Ω̄1."""
    assert geo.on_D2_omega1(5.0, 0.0, 0.0)
    assert not geo.on_D2_omega2(5.0, 0.0, 0.0)


def test_D2_omega2_inlet(geo):
    """r0 < r < r1, z=0 debe estar en D2∩Ω̄2 (entrada fluido)."""
    r = (geo.r0 + geo.r1) / 2
    assert geo.on_D2_omega2(r, 0.0, 0.0)
    assert not geo.on_D2_omega1(r, 0.0, 0.0)


def test_D2_omega3_outlet(geo):
    """r < r0, z=0 debe estar en D2∩Ω̄3 (salida fluido)."""
    assert geo.on_D2_omega3(geo.r0 / 2, 0.0, 0.0)


def test_Gamma_ext(geo):
    """r=R, 0 < z < h debe estar en Γ_ext."""
    assert geo.on_Gamma_ext(geo.R, 0.0, geo.h / 2)
    assert not geo.on_Gamma_ext(geo.R, 0.0, 0.0)  # z=0 es D2


# ---------------------------------------------------------------------------
# Tests de etiquetado
# ---------------------------------------------------------------------------

def test_label_cells_vectorized(geo):
    """label_cells retorna etiquetas correctas para múltiples puntos."""
    centroids = np.array([
        [5.0, 0.0, 50.0],        # Ω1 corona
        [0.0, 0.0, 175.0],       # Ω1 tapón inferior
        [(geo.r0+geo.r1)/2, 0.0, geo.L/2],  # Ω2
        [geo.r0/2, 0.0, geo.L/2],           # Ω3
    ])
    labels = geo.label_cells(centroids)
    assert labels[0] == 1
    assert labels[1] == 1
    assert labels[2] == 2
    assert labels[3] == 3
