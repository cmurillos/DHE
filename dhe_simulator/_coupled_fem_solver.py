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
        orig_tol = geo.tol
        import dataclasses
        geo_loose = dataclasses.replace(geo, tol=orig_tol * factor)
        # Obtener el mismo método en geo_loose
        method_name = pred_fn.__func__.__name__
        return getattr(geo_loose, method_name)(x, y, z)

    return loose


# ---------------------------------------------------------------------------
# Utilidad: construir submalla de un subdominio
# ---------------------------------------------------------------------------

def _make_submesh(global_nodes, global_elements, elem_subset_idx):
    """
    Construye un MeshTet local a partir de un subconjunto de elementos globales.
    Filtra tetraedros con volumen < tol_vol (degenerados) para evitar NaN
    en el ensamblaje FEM.

    Retorna
    -------
    sub_mesh : MeshTet
    node_global_to_local : dict {global_idx: local_idx}
    node_local_to_global : ndarray (N_local,)  índices en global_nodes
    """
    sub_elems_global = global_elements[elem_subset_idx]  # (K, 4)

    # Filtrar tetraedros degenerados (volumen < 1e-20)
    pts = global_nodes[sub_elems_global]    # (K, 4, 3)
    v1 = pts[:, 1, :] - pts[:, 0, :]
    v2 = pts[:, 2, :] - pts[:, 0, :]
    v3 = pts[:, 3, :] - pts[:, 0, :]
    vols = np.abs(np.einsum('ij,ij->i', v1, np.cross(v2, v3))) / 6.0
    good = vols > 1e-14  # Filtrar tetraedros casi-degenerados (coplanares)
    if not np.all(good):
        n_bad = (~good).sum()
        warnings.warn(f"Submalla: {n_bad} tetraedros degenerados eliminados.")
        sub_elems_global = sub_elems_global[good]

    if len(sub_elems_global) == 0:
        raise RuntimeError("Submalla vacía tras filtrar tetraedros degenerados.")

    node_local_to_global = np.unique(sub_elems_global.ravel())
    node_global_to_local = {g: l for l, g in enumerate(node_local_to_global)}

    sub_nodes = global_nodes[node_local_to_global]  # (N_local, 3)
    sub_elems_local = np.vectorize(node_global_to_local.get)(sub_elems_global)

    sub_mesh = MeshTet(sub_nodes.T, sub_elems_local.T, validate=False)
    return sub_mesh, node_global_to_local, node_local_to_global


class CoupledCylinderFEMSolver:
    """
    Solver FEM acoplado para el sistema térmico DHE.

    Parameters
    ----------
    coupled_mesh : CoupledCylinderMesh
    geometry : DHEGeometry
    """

    def __init__(self, coupled_mesh, geometry):
        self.mesh_obj = coupled_mesh
        self.geometry = geometry

        global_nodes = coupled_mesh.nodes       # (N_global, 3)
        global_elements = coupled_mesh.elements  # (N_tets, 4)
        cell_sub = coupled_mesh.cell_subdomains

        for name in ('omega1', 'omega2', 'omega3'):
            if name not in cell_sub:
                raise RuntimeError(f"Subdominio '{name}' ausente en la malla")

        elem = ElementTetP1()

        # Construir submallas y bases independientes
        self._sub = {}
        for name, idx in cell_sub.items():
            mesh_s, g2l, l2g = _make_submesh(global_nodes, global_elements, idx)
            basis_s = Basis(mesh_s, elem)
            self._sub[name] = {
                'mesh': mesh_s,
                'basis': basis_s,
                'g2l': g2l,
                'l2g': l2g,
                'elem_global_idx': idx,
            }

        self.basis1 = self._sub['omega1']['basis']
        self.basis2 = self._sub['omega2']['basis']
        self.basis3 = self._sub['omega3']['basis']

        self.N1 = self.basis1.N
        self.N2 = self.basis2.N
        self.N3 = self.basis3.N
        self.N_total = self.N1 + self.N2 + self.N3

        # Coordenadas de DOFs locales (N_local, 3)
        self.coords1 = self.basis1.doflocs.T
        self.coords2 = self.basis2.doflocs.T
        self.coords3 = self.basis3.doflocs.T

        # KD-trees para búsqueda de DOFs por coordenada física
        self._tree1 = cKDTree(self.coords1)
        self._tree2 = cKDTree(self.coords2)
        self._tree3 = cKDTree(self.coords3)

        # KD-tree del mesh global para reconstrucción de campo nodal
        self._tree_global = cKDTree(global_nodes)

    # ------------------------------------------------------------------
    # Utilidades de coeficientes
    # ------------------------------------------------------------------

    def _interp_field(self, basis, func):
        """Evalúa func en DOFs de basis y retorna DiscreteField."""
        c = basis.doflocs.T
        vals = func(c[:, 0], c[:, 1], c[:, 2])
        return basis.interpolate(np.asarray(vals, dtype=float))

    def _interp_scalar(self, basis, val):
        return basis.interpolate(np.full(basis.N, float(val)))

    # ------------------------------------------------------------------
    # Robin sobre facets de una frontera (desde la perspectiva de basis)
    # ------------------------------------------------------------------

    def _get_boundary_facets(self, basis, name):
        """
        Retorna los índices de facets de frontera del mesh local de basis
        que corresponden a la frontera global `name`.

        DECISIÓN sobre detección: se usan los NODOS de cada facet (no el centroide)
        para determinar si el facet pertenece a una frontera. Un facet pertenece a
        la frontera `name` si TODOS SUS VÉRTICES satisfacen el predicado geométrico
        con una tolerancia ampliada (×50). Esto es necesario porque el centroide de
        un triángulo inscrito en un cilindro NO cae sobre el cilindro en general
        (el radio del centroide es siempre ≤ al radio de los vértices).
        """
        geo = self.geometry
        mesh = basis.mesh

        bfacets = mesh.boundary_facets()
        if len(bfacets) == 0:
            return np.array([], dtype=int)

        face_nodes = mesh.facets[:, bfacets]   # (3, N_bf)

        predicates = {
            'C1':        geo.on_C1,
            'C2':        geo.on_C2,
            'C_L2':      geo.on_C_L2,
            'C_L3':      geo.on_C_L3,
            'D1':        geo.on_D1,
            'D2_omega1': geo.on_D2_omega1,
            'D2_omega2': geo.on_D2_omega2,
            'D2_omega3': geo.on_D2_omega3,
            'Gamma_ext': geo.on_Gamma_ext,
        }
        pred = predicates.get(name)
        if pred is None:
            return np.array([], dtype=int)

        # Usar tolerancia ampliada para verificar cada vértice
        pred_loose = _make_loose_predicate(pred, factor=50)

        # Un facet está en la frontera si TODOS sus 3 vértices satisfacen pred
        all_nodes_flat = face_nodes.ravel()  # (3*N_bf,)
        pts = mesh.p[:, all_nodes_flat].T     # (3*N_bf, 3)
        in_boundary = pred_loose(pts[:, 0], pts[:, 1], pts[:, 2]).reshape(3, -1)
        mask = in_boundary.all(axis=0)         # (N_bf,)
        return bfacets[mask]

    def _robin_bilinear(self, basis, name, delta):
        """∫_{name} δ u v dS → sparse (N, N)."""
        facets = self._get_boundary_facets(basis, name)
        if len(facets) == 0:
            return sp.csc_matrix((basis.N, basis.N))
        fb = FacetBasis(basis.mesh, ElementTetP1(), facets=facets)
        delta_interp = fb.interpolate(np.full(fb.N, delta))
        return asm(robin_surface, fb, delta=delta_interp)

    def _robin_linear(self, basis, name, delta, T_bc):
        """∫_{name} δ T_bc v dS → ndarray (N,)."""
        facets = self._get_boundary_facets(basis, name)
        if len(facets) == 0:
            return np.zeros(basis.N)
        fb = FacetBasis(basis.mesh, ElementTetP1(), facets=facets)
        delta_interp = fb.interpolate(np.full(fb.N, delta))
        if callable(T_bc):
            c = fb.doflocs.T
            T_vals = np.asarray(T_bc(c[:, 0], c[:, 1], c[:, 2]), dtype=float)
        else:
            T_vals = np.full(fb.N, float(T_bc))
        T_interp = fb.interpolate(T_vals)
        return asm(rhs_surface, fb, delta=delta_interp, T_bc=T_interp)

    # ------------------------------------------------------------------
    # Bloques cruzados entre subdominios
    # ------------------------------------------------------------------

    def _cross_block(self, interface_name, basis_test, basis_trial, delta):
        """
        B[i,j] = ∫_{interface} δ φ^trial_j φ^test_i dS
        → sparse (N_test, N_trial)

        Identifica la interfaz como los facets de frontera del mesh local de
        basis_test que caen en 'interface_name', luego mapea los nodos a los
        DOFs de basis_trial por coordenada física.
        """
        facets = self._get_boundary_facets(basis_test, interface_name)
        if len(facets) == 0:
            return sp.csc_matrix((basis_test.N, basis_trial.N))

        mesh_t = basis_test.mesh
        face_nodes_local = mesh_t.facets[:, facets]  # (3, N_f)

        tree_trial = cKDTree(basis_trial.doflocs.T)
        tol = self.geometry.tol * 1000

        N_test = basis_test.N
        N_trial = basis_trial.N

        # Puntos de cuadratura P1 triangular (3 puntos, orden 2)
        xi_pts = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        w_pts = np.array([1/6, 1/6, 1/6])

        rows, cols, data = [], [], []

        for fi in range(face_nodes_local.shape[1]):
            n0, n1, n2 = face_nodes_local[:, fi]
            p0 = mesh_t.p[:, n0]
            p1 = mesh_t.p[:, n1]
            p2 = mesh_t.p[:, n2]

            v1 = p1 - p0
            v2 = p2 - p0
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            if area < 1e-20:
                continue

            verts = np.array([p0, p1, p2])

            # DOFs de basis_test (son los nodos locales del mesh_t)
            dof_test = np.array([n0, n1, n2])  # índices locales de DOFs P1

            # DOFs de basis_trial (localizar por coordenada física)
            d_r, dof_trial = tree_trial.query(verts, distance_upper_bound=tol)
            valid_r = d_r < tol

            if not np.any(valid_r):
                continue

            for qi, ((xi, eta), w) in enumerate(zip(xi_pts, w_pts)):
                phi = np.array([1 - xi - eta, xi, eta])
                contrib = 2.0 * area * w * delta

                for i_loc in range(3):
                    di = dof_test[i_loc]
                    if di >= N_test:
                        continue
                    for j_loc in range(3):
                        if not valid_r[j_loc]:
                            continue
                        dj = dof_trial[j_loc]
                        if dj >= N_trial:
                            continue
                        rows.append(di)
                        cols.append(dj)
                        data.append(contrib * phi[i_loc] * phi[j_loc])

        if not data:
            return sp.csc_matrix((N_test, N_trial))

        return sp.coo_matrix((data, (rows, cols)),
                             shape=(N_test, N_trial)).tocsc()

    # ------------------------------------------------------------------
    # Ensamblaje del sistema global
    # ------------------------------------------------------------------

    def assemble_system(
        self,
        rho_func, c_func, k_func,
        alpha2, alpha3, U_z,
        delta_0, delta_1, delta_2, delta_f,
        T_f,
    ):
        """
        Ensambla M y A del sistema acoplado por bloques.

        M = block_diag(M1, M2, M3)
        A = [[A11, A12,  0  ],
             [A21, A22,  A23],
             [ 0,  A32,  A33]]
        """
        print("  Ensamblando Ω1...", flush=True)
        M1, A11, L_D1 = self._assemble_omega1(
            rho_func, c_func, k_func, delta_0, delta_1, delta_f, T_f
        )

        print("  Ensamblando Ω2...", flush=True)
        M2, A22 = self._assemble_omega2(alpha2, U_z, delta_1, delta_2)

        print("  Ensamblando Ω3...", flush=True)
        M3, A33 = self._assemble_omega3(alpha3, U_z, delta_2)

        print("  Ensamblando bloques cruzados...", flush=True)
        A12 = -self._cross_block('C1', self.basis1, self.basis2, delta_1)
        A21 = -self._cross_block('C1', self.basis2, self.basis1, delta_1)
        A23 = -self._cross_block('C2', self.basis2, self.basis3, delta_2)
        A32 = -self._cross_block('C2', self.basis3, self.basis2, delta_2)

        N1, N2, N3 = self.N1, self.N2, self.N3

        self.M = sp.block_diag([M1, M2, M3], format='csc')
        self.A = sp.bmat([
            [A11,                      A12,                      sp.csc_matrix((N1, N3))],
            [A21,                      A22,                      A23],
            [sp.csc_matrix((N3, N1)), A32,                      A33],
        ], format='csc')

        self.L_const = np.zeros(self.N_total)
        if L_D1 is not None:
            self.L_const[:N1] += L_D1

        self.delta_0 = delta_0
        self._T_f = T_f
        print(f"  Listo. M={self.M.shape}, A={self.A.shape}", flush=True)

    def _assemble_omega1(self, rho_func, c_func, k_func,
                         delta_0, delta_1, delta_f, T_f):
        b = self.basis1
        rho_c_interp = self._interp_field(b, lambda x, y, z: rho_func(x, y, z) * c_func(x, y, z))
        k_interp = self._interp_field(b, k_func)

        M1 = asm(mass_rho_c, b, rho_c=rho_c_interp)
        A11 = asm(diffusion_k, b, k=k_interp)
        A11 += self._robin_bilinear(b, 'D2_omega1', delta_0)
        A11 += self._robin_bilinear(b, 'C1', delta_1)
        A11 += self._robin_bilinear(b, 'D1', delta_f)
        L_D1 = self._robin_linear(b, 'D1', delta_f, T_f)
        return M1, A11, L_D1

    def _assemble_omega2(self, alpha2, U_z, delta_1, delta_2):
        b = self.basis2
        U_interp = self._interp_scalar(b, U_z)
        M2 = asm(mass, b)
        A22 = alpha2 * asm(laplace, b)
        A22 += asm(advection_pos_z, b, U_z=U_interp)
        A22 += self._robin_bilinear(b, 'C1', delta_1)
        A22 += self._robin_bilinear(b, 'C2', delta_2)
        return M2, A22

    def _assemble_omega3(self, alpha3, U_z, delta_2):
        b = self.basis3
        U_interp = self._interp_scalar(b, U_z)
        M3 = asm(mass, b)
        A33 = alpha3 * asm(laplace, b)
        A33 += asm(advection_neg_z, b, U_z=U_interp)
        A33 += self._robin_bilinear(b, 'C2', delta_2)
        return M3, A33

    # ------------------------------------------------------------------
    # Forma lineal con Ts(t, x, y)
    # ------------------------------------------------------------------

    def _assemble_L_Ts(self, Ts_func, t):
        """δ0 ∫_{D2∩Ω̄1} Ts(t,·) v dS."""
        L = self.L_const.copy()

        def Ts_at_t(x, y, z):
            return Ts_func(t, x, y)

        L_Ts = self._robin_linear(self.basis1, 'D2_omega1', self.delta_0, Ts_at_t)
        L[:self.N1] += L_Ts
        return L

    # ------------------------------------------------------------------
    # DOFs de Dirichlet y restricciones de igualdad
    # ------------------------------------------------------------------

    def _get_dirichlet_info(self, T_int):
        """
        Retorna:
            dofs_inlet : ndarray  DOFs de T2 en D2∩Ω̄2 (offset N1 incluido)
            slave_dofs : ndarray  T2|CL2 y T3|CL3 (offsets incluidos)
            master_dofs : ndarray T1|CL2 y T1|CL3 correspondientes
        """
        geo = self.geometry
        N1, N2 = self.N1, self.N2
        tol = geo.tol * 1000

        # Entrada del fluido
        x2, y2, z2 = self.coords2[:, 0], self.coords2[:, 1], self.coords2[:, 2]
        inlet_local = np.where(geo.on_D2_omega2(x2, y2, z2))[0]
        dofs_inlet = inlet_local + N1

        # T2|CL2 ↔ T1|CL2
        x1, y1, z1 = self.coords1[:, 0], self.coords1[:, 1], self.coords1[:, 2]
        m1_CL2 = np.where(geo.on_C_L2(x1, y1, z1))[0]
        m2_CL2 = np.where(geo.on_C_L2(x2, y2, z2))[0]
        s_CL2, m_CL2 = self._match_dofs(self.coords2[m2_CL2], self.coords1[m1_CL2],
                                         m2_CL2 + N1, m1_CL2, tol)

        # T3|CL3 ↔ T1|CL3
        x3, y3, z3 = self.coords3[:, 0], self.coords3[:, 1], self.coords3[:, 2]
        m1_CL3 = np.where(geo.on_C_L3(x1, y1, z1))[0]
        m3_CL3 = np.where(geo.on_C_L3(x3, y3, z3))[0]
        s_CL3, m_CL3 = self._match_dofs(self.coords3[m3_CL3], self.coords1[m1_CL3],
                                         m3_CL3 + N1 + N2, m1_CL3, tol)

        slave_dofs = np.concatenate([s_CL2, s_CL3])
        master_dofs = np.concatenate([m_CL2, m_CL3])
        return dofs_inlet, slave_dofs, master_dofs

    @staticmethod
    def _match_dofs(coords_slave, coords_master, global_slave, global_master, tol):
        if len(coords_slave) == 0 or len(coords_master) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        tree = cKDTree(coords_master)
        dist, idx = tree.query(coords_slave, distance_upper_bound=tol * 10)
        valid = dist < tol * 10
        return global_slave[valid], global_master[idx[valid]]

    # ------------------------------------------------------------------
    # Imposición de Dirichlet
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_dirichlet(MA_lil, rhs, dofs, vals):
        """Eliminación de DOFs Dirichlet en matriz lil_matrix."""
        for dof, val in zip(dofs, vals):
            col = np.asarray(MA_lil.getcol(dof).todense()).ravel()
            rhs -= col * val
            MA_lil[dof, :] = 0
            MA_lil[:, dof] = 0
            MA_lil[dof, dof] = 1.0
            rhs[dof] = val

    # ------------------------------------------------------------------
    # Vector inicial
    # ------------------------------------------------------------------

    def _initial_vector(self, T0_func):
        def ev(coords):
            return T0_func(coords[:, 0], coords[:, 1], coords[:, 2])
        return np.concatenate([ev(self.coords1), ev(self.coords2), ev(self.coords3)])

    # ------------------------------------------------------------------
    # Estabilización
    # ------------------------------------------------------------------

    def stabilize(self, T0_vec, dt, Ts_const=None, tol=1e-6, max_iter=50_000):
        """
        Itera sin advección hasta ||ΔT||/||T|| < tol.
        Retorna (T_stable, n_iter).
        """
        from ._surface_temperature import SurfaceTemperature
        if Ts_const is None:
            Ts_const = SurfaceTemperature.constant(0.0)

        T = T0_vec.copy()
        dofs_inlet, slave_dofs, master_dofs = self._get_dirichlet_info(T_int=0.0)
        MA_base = (self.M + dt * self.A)

        print("  Estabilizando...", flush=True)
        t0 = time.time()

        for n in range(max_iter):
            L_n = self._assemble_L_Ts(Ts_const, t=0.0)
            MA = MA_base.tolil()
            rhs = self.M @ T + dt * L_n

            if len(slave_dofs) > 0:
                self._apply_dirichlet(MA, rhs, slave_dofs, T[master_dofs])

            T_new = spla.spsolve(MA.tocsc(), rhs)
            rel_err = np.linalg.norm(T_new - T) / (np.linalg.norm(T) + 1e-30)
            T = T_new

            if n % 5000 == 0:
                print(f"    iter {n:6d} | err_rel={rel_err:.2e} | "
                      f"cpu={time.time()-t0:.1f}s", flush=True)
            if rel_err < tol:
                print(f"  Convergió en {n} iters.", flush=True)
                return T, n

        warnings.warn("stabilize() no convergió.")
        return T, max_iter

    # ------------------------------------------------------------------
    # Resolución temporal
    # ------------------------------------------------------------------

    def solve(
        self,
        T0_func,
        dt, tf, t_save,
        Ts_func,
        T_int=285.0,
        t_on=0.0, t_off=None,
        stabilize_first=True,
    ):
        """
        Integra de t=0 a t=tf con Euler implícito.
        Retorna DHEResult.
        """
        T = self._initial_vector(T0_func)

        if stabilize_first:
            T, _ = self.stabilize(T, dt, Ts_const=Ts_func)

        dofs_inlet, slave_dofs, master_dofs = self._get_dirichlet_info(T_int)

        times_saved, T_saved = [], []
        n_steps = int(np.ceil(tf / dt))
        next_save = t_save

        print(f"  Simulación: {n_steps} pasos, dt={dt:.0f}s, tf={tf:.0f}s", flush=True)
        t_cpu = time.time()

        for step in range(n_steps):
            t_new = (step + 1) * dt
            active = (t_new >= t_on) and (t_off is None or t_new <= t_off)

            L_n = self._assemble_L_Ts(Ts_func, t_new)
            MA = (self.M + dt * self.A).tolil()
            rhs = self.M @ T + dt * L_n

            if len(slave_dofs) > 0:
                self._apply_dirichlet(MA, rhs, slave_dofs, T[master_dofs])

            if active and len(dofs_inlet) > 0:
                self._apply_dirichlet(MA, rhs, dofs_inlet,
                                      np.full(len(dofs_inlet), T_int))

            T = spla.spsolve(MA.tocsc(), rhs)

            if t_new >= next_save - dt * 1e-6:
                times_saved.append(t_new)
                T_saved.append(T.copy())
                next_save += t_save
                print(f"  t={t_new:.0f}s | paso {step+1}/{n_steps} | "
                      f"T_mean={T.mean():.2f}K | cpu={time.time()-t_cpu:.1f}s",
                      flush=True)

        T_node = self._reconstruct_node_field(np.array(T_saved))
        geo = self.geometry
        mesh_obj = self.mesh_obj
        return DHEResult(
            nodes=mesh_obj.nodes,
            elements=mesh_obj.elements,
            boundary_faces=mesh_obj.boundary_faces,
            cell_tags=mesh_obj.cell_tags,
            facet_tags=mesh_obj.facet_tags,
            times=np.array(times_saved),
            T=T_node,
            geometry_params={'r0': geo.r0, 'r1': geo.r1, 'R': geo.R,
                             'L': geo.L, 'h': geo.h},
        )

    # ------------------------------------------------------------------
    # Reconstrucción de campo nodal desde DOFs de submallas
    # ------------------------------------------------------------------

    def _reconstruct_node_field(self, T_block_array):
        """
        T_block_array: (N_snaps, N1+N2+N3) — DOFs de submallas
        Retorna: (N_snaps, N_global) — campo en nodos globales

        Usa KD-tree global. Para nodos compartidos entre subdominios
        (e.g., en C1) se promedian los valores de los distintos campos.
        """
        N_snaps = T_block_array.shape[0]
        N_global = self.mesh_obj.nodes.shape[0]
        T_sum = np.zeros((N_snaps, N_global))
        T_cnt = np.zeros(N_global, dtype=int)

        def accumulate(coords, offset, count):
            _, idx = self._tree_global.query(coords)
            unique_idx = np.unique(idx)
            for s in range(N_snaps):
                np.add.at(T_sum[s], idx,
                          T_block_array[s, offset:offset + len(coords)])
            np.add.at(T_cnt, idx, 1)

        accumulate(self.coords1, 0, self.N1)
        accumulate(self.coords2, self.N1, self.N2)
        accumulate(self.coords3, self.N1 + self.N2, self.N3)

        # Promedio (nodos con múltiples asignaciones son interfaces)
        T_cnt_safe = np.where(T_cnt > 0, T_cnt, 1)
        T_node = T_sum / T_cnt_safe[np.newaxis, :]

        # Nodos no asignados (raro, pero por robustez)
        unassigned = T_cnt == 0
        if np.any(unassigned):
            valid = ~unassigned
            tree_v = cKDTree(self.mesh_obj.nodes[valid])
            _, nn = tree_v.query(self.mesh_obj.nodes[unassigned])
            for s in range(N_snaps):
                T_node[s, unassigned] = T_node[s, valid][nn]

        return T_node
