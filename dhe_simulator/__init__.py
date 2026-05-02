"""
dhe_simulator v1.0.0 — Coupled thermal simulation of DHE systems.

Resuelve el sistema acoplado de tres PDEs en dominios cilíndricos
concéntricos con condición de frontera superior alimentada por imágenes
térmicas de dron.

Stack: numpy + scipy + scikit-fem (sin FEniCS, dolfin ni FEniCSx).

Uso rápido
----------
    from dhe_simulator import DHE_simulation, SurfaceTemperature

    sim = DHE_simulation(
        csv='perfiles_paramo.csv',
        r0=0.05, r1=0.10, R=20.0,
        L=150.0, h=200.0,
        alpha2=1.4e-7, alpha3=1.4e-7, U_z=0.5,
        delta_0=0.1, delta_1=10.0, delta_2=0.5, delta_f=1.0,
        T_int=285.0, T_f=355.0,
        Ts=SurfaceTemperature.constant(285.0),
    )
    result = sim.run(dt=3600, t_save=86400, tf=2592000)
    result.save('out.npz')
    result.export_xdmf('out')
"""

from .dhe_simulation import DHE_simulation
from ._surface_temperature import SurfaceTemperature
from ._result import DHEResult
from ._geometry import DHEGeometry

__all__ = ['DHE_simulation', 'SurfaceTemperature', 'DHEResult', 'DHEGeometry']
__version__ = '1.0.0'
