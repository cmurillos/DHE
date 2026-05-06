"""
ts_from_drone_demo.py
=====================

Ejemplo con Ts variable en el tiempo usando frames sintéticos de dron.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dhe_simulator import DHE_simulation, SurfaceTemperature

# Crear frames sintéticos de dron: variación diurna simple
# Cada frame: (t_i, X_2d, Y_2d, T_2d) sobre la superficie z=0
frames = []
x_grid = np.linspace(-20, 20, 20)
y_grid = np.linspace(-20, 20, 20)
X, Y = np.meshgrid(x_grid, y_grid)
for hour in range(0, 25, 6):
    t_i = hour * 3600.0
    # Temperatura diurna: 283 K de noche, +5 K al mediodía
    T_i = 283 + 5 * np.sin(np.pi * hour / 12) * np.exp(-(X**2 + Y**2) / 200)
    frames.append((t_i, X, Y, T_i))

Ts = SurfaceTemperature.from_drone_frames(frames, method='nearest')

sim = DHE_simulation(
    csv='examples/perfiles_paramo_ruiz_tolima.csv',
    r0=0.05, r1=0.10, R=20.0,
    L=150.0, h=200.0,
    alpha2=1.4e-7, alpha3=1.4e-7,
    U_z=0.01,
    delta_0=0.5, delta_1=10.0, delta_2=0.5, delta_f=1.0,
    T_int=285.0, T_f=355.0,
    Ts=Ts,
    nr_omega1=5, nr_omega2=3, nr_omega3=3,
    nz_above_L=10, nz_below_L=5, ntheta=8,
    tries=1, seed=0,
)

result = sim.run(
    dt=1800,
    t_save=3600 * 6,
    tf=3600 * 48,
    stabilize_first=False,
    output_prefix='drone_demo_out',
)

print(f"T media final: {result.T[-1].mean():.2f} K")
