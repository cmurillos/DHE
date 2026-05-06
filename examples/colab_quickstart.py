"""
colab_quickstart.py
===================

Ejemplo mínimo: simula 30 días con condiciones constantes.
Ejecutar desde la raíz del repo:
    python examples/colab_quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dhe_simulator import DHE_simulation, SurfaceTemperature

sim = DHE_simulation(
    csv='examples/perfiles_paramo_ruiz_tolima.csv',
    r0=0.05, r1=0.10, R=20.0,
    L=150.0, h=200.0,
    alpha2=1.4e-7, alpha3=1.4e-7,
    U_z=0.01,
    delta_0=0.1, delta_1=10.0, delta_2=0.5, delta_f=1.0,
    T_int=285.0, T_f=355.0,
    Ts=SurfaceTemperature.constant(285.0),
    nr_omega1=5, nr_omega2=3, nr_omega3=3,
    nz_above_L=10, nz_below_L=5, ntheta=8,
    tries=1, seed=42,
)

result = sim.run(
    dt=3600,           # 1 hora
    t_save=86400,      # guardar cada día
    tf=86400 * 30,     # 30 días
    stabilize_first=False,
    output_prefix='quickstart_out',
)

print(f"\nSimulación completada.")
print(f"Snapshots: {len(result.times)}")
print(f"T media final: {result.T[-1].mean():.2f} K")

result.export_xdmf('quickstart_out')
print("Exportado quickstart_out.xdmf + quickstart_out.h5")
