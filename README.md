# dhe_simulator v1.0.0

Simulación térmica acoplada de un sistema **Downhole Heat Exchanger (DHE)** en circuito cerrado.

## Modelo matemático

Sistema acoplado de 3 PDEs sobre 3 subdominios cilíndricos concéntricos:

- **Ω1** (subsuelo): conducción heterogénea con coeficientes ρ(x), c(x), k(x)
- **Ω2** (anular descendente): advección-difusión con flujo +z
- **Ω3** (interior ascendente): advección-difusión con flujo -z

Condiciones Robin en todas las interfaces (C1, C2, D1, D2) y Dirichlet en la entrada del fluido (D2 ∩ Ω̄2).

## Stack tecnológico

- `numpy`, `scipy`, `pandas` — álgebra y datos
- `scikit-fem` — FEM (sin FEniCS, dolfin ni FEniCSx)
- `matplotlib` — visualización
- `h5py` — exportación XDMF/HDF5 para ParaView

## Instalación

```bash
pip install -r requirements.txt
pip install -e .
```

## Uso rápido

```python
from dhe_simulator import DHE_simulation, SurfaceTemperature

sim = DHE_simulation(
    csv='examples/perfiles_paramo_ruiz_tolima.csv',
    r0=0.05, r1=0.10, R=20.0,
    L=150.0, h=200.0,
    alpha2=1.4e-7, alpha3=1.4e-7, U_z=0.01,
    delta_0=0.1, delta_1=10.0, delta_2=0.5, delta_f=1.0,
    T_int=285.0, T_f=355.0,
    Ts=SurfaceTemperature.constant(285.0),
)
result = sim.run(dt=3600, t_save=86400, tf=2592000)
result.save('out.npz')
result.export_xdmf('out')
```

## Estructura

```
dhe_simulator/
├── __init__.py
├── dhe_simulation.py          # Clase orquestadora
├── _geometry.py               # Predicados de subdominio y frontera
├── _coupled_mesh.py           # Malla tetraédrica conforme
├── _coupled_fem_solver.py     # Ensamblaje FEM y resolución temporal
├── _generador_campos.py       # Campos heterogéneos desde CSV
├── _surface_temperature.py    # Ts(t, x, y) desde constante/callable/dron
└── _result.py                 # DHEResult con save/load/export_xdmf
examples/
├── colab_quickstart.py
├── ts_from_drone_demo.py
└── perfiles_paramo_ruiz_tolima.csv
tests/
└── test_geometry.py
```

## Tests

```bash
pytest tests/
```

## Decisiones de implementación

- **Una sola malla global conforme** con tags de subdominio.
- **Bases por subdominio** (DOFs separados para T1, T2, T3), bloques cruzados B12/B21/B23/B32 ensamblados sobre interfaces C1, C2.
- **Euler implícito** para estabilidad ante difusión dominante.
- **Imposición de Dirichlet por eliminación de DOFs** (no penalización).
- **δ_L → eliminación fuerte**: las igualdades T2=T1, T3=T1 en C_L se imponen como restricciones nodales, no como términos débiles.
- **Robin en D1** con T_f configurable (poner `delta_f=0` para flujo nulo).
