"""
_surface_temperature.py
=======================

Provee Ts: D2 × ℝ+ → ℝ — la temperatura superficial que actúa como
condición Robin sobre D2 ∩ Ω̄1.

Tres modos:
    1) Constante: SurfaceTemperature.constant(285.0)
    2) Callable: SurfaceTemperature.from_callable(lambda t, x, y: ...)
    3) Frames de dron: SurfaceTemperature.from_drone_frames(frames)
"""

import numpy as np


class SurfaceTemperature:
    """Encapsula Ts(t, x, y) con varios constructores."""

    def __init__(self, eval_func):
        """
        Parameters
        ----------
        eval_func : callable(t, x, y) -> ndarray
        """
        self._eval = eval_func

    def __call__(self, t, x, y):
        x, y = np.asarray(x), np.asarray(y)
        return np.asarray(self._eval(t, x, y), dtype=float)

    @classmethod
    def constant(cls, T_const):
        """Ts = T_const en todo tiempo y posición."""
        T_const = float(T_const)

        def _eval(t, x, y):
            return np.full(np.shape(x), T_const)

        return cls(_eval)

    @classmethod
    def from_callable(cls, func):
        """Acepta func(t, x, y) -> array_like y lo envuelve."""
        return cls(func)

    @classmethod
    def from_drone_frames(cls, frames, method='nearest'):
        """
        Parameters
        ----------
        frames : list[tuple(float, ndarray_2d, ndarray_2d, ndarray_2d)]
            Lista de (t_i, X_i, Y_i, T_i) donde X_i, Y_i, T_i son
            arrays 2D que representan el raster térmico en z=0.
        method : {'nearest', 'bilinear'}
            Método de interpolación espacial.

        Interpolación temporal: piecewise-constant (frame más reciente con t_i ≤ t).
        """
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree

        # Ordenar por tiempo
        frames_sorted = sorted(frames, key=lambda f: f[0])
        times = [f[0] for f in frames_sorted]

        # Pre-construir estructuras de interpolación por frame
        interp_data = []
        for (t_i, X_i, Y_i, T_i) in frames_sorted:
            pts = np.column_stack([X_i.ravel(), Y_i.ravel()])
            vals = T_i.ravel()
            if method == 'nearest':
                tree = cKDTree(pts)
                interp_data.append(('nearest', tree, vals))
            else:
                interp_data.append(('bilinear', pts, vals))

        def _eval(t, x, y):
            x, y = np.asarray(x), np.asarray(y)
            # Seleccionar el frame con t_i más reciente ≤ t
            idx = 0
            for i, ti in enumerate(times):
                if ti <= t:
                    idx = i
                else:
                    break

            mode, *data = interp_data[idx]
            query_pts = np.column_stack([x.ravel(), y.ravel()])

            if mode == 'nearest':
                tree, vals = data
                _, nn_idx = tree.query(query_pts)
                result = vals[nn_idx]
            else:
                pts, vals = data
                result = griddata(pts, vals, query_pts, method='linear',
                                  fill_value=np.nanmean(vals))
            return result.reshape(np.shape(x))

        return cls(_eval)
