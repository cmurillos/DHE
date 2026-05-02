"""
DHEMesh — Coaxial Deep Borehole Heat Exchanger mesh generator.

Stack: Gmsh (Python API) + meshio + DOLFINx
"""

import math
import os
import tempfile

import gmsh
import meshio

from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl


class DHEMesh:
    """
    Builds a conforming 3D mesh of the coaxial DHE, tags every subdomain
    and interface, and exposes integration measures for variational assembly.

    Domain decomposition
    --------------------
    Ω1 (tag=1)  Geological subsurface  { r1<r<R, 0<z<h } ∪ { r<r1, L<z<h }
    Ω2 (tag=2)  Annular channel        { r0<r<r1, 0<z<L }
    Ω3 (tag=3)  Inner tube             { r<r0,    0<z<L }

    External boundary tags (ds)
    ---------------------------
    10  D2_rock    z=0, r∈[r1, R]
    11  D2_inlet   z=0, r∈[r0, r1]
    12  D2_outlet  z=0, r∈[0,  r0]
    13  D1_Gext    z=h  ∪  r=R

    Internal interface tags (dS)
    ----------------------------
    20  C1   r=r1, z∈(0,L)
    21  C2   r=r0, z∈(0,L)
    22  CL   z=L,  r∈[0,r1]
    """

    # ── subdomain tags ─────────────────────────────────────────────
    TAG_OMEGA1 = 1
    TAG_OMEGA2 = 2
    TAG_OMEGA3 = 3

    # ── boundary tags ──────────────────────────────────────────────
    TAG_D2_ROCK   = 10
    TAG_D2_INLET  = 11
    TAG_D2_OUTLET = 12
    TAG_D1_GEXT   = 13
    TAG_C1        = 20
    TAG_C2        = 21
    TAG_CL        = 22

    def __init__(
        self,
        r0, r1, R, L, h,
        n_r3, n_r2, n_r1, n_theta,
        n_z_active, n_z_deep,
    ):
        assert 0 < r0 < r1 < R, "Radial constraint violated: 0 < r0 < r1 < R"
        assert 0 < L < h,       "Axial constraint violated: 0 < L < h"

        self._r0, self._r1, self._R = r0, r1, R
        self._L,  self._h           = L,  h

        self._workdir = tempfile.mkdtemp(prefix="dhe_")
        msh_path    = os.path.join(self._workdir, "dhe.msh")
        cells_xdmf  = os.path.join(self._workdir, "dhe_cells.xdmf")
        facets_xdmf = os.path.join(self._workdir, "dhe_facets.xdmf")

        self._n = dict(
            n_r3=n_r3, n_r2=n_r2, n_r1=n_r1,
            n_theta=n_theta, n_z_active=n_z_active, n_z_deep=n_z_deep,
        )
        self._build_gmsh(
            r0, r1, R, L, h,
            n_r3, n_r2, n_r1, n_theta, n_z_active, n_z_deep,
            msh_path,
        )
        self._convert_meshio(msh_path, cells_xdmf, facets_xdmf)
        self._load_dolfinx(cells_xdmf, facets_xdmf)
        self._create_measures()

    # ════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ════════════════════════════════════════════════════════════════

    def get_measures(self):
        """Return (dx, ds, dS, n) for use in external form assembly."""
        return self.dx, self.ds, self.dS, self.n

    def get_tags(self):
        """Return (cell_tags, facet_tags) for boundary condition setup."""
        return self.cell_tags, self.facet_tags

    # ════════════════════════════════════════════════════════════════
    # Gmsh geometry and mesh generation
    # ════════════════════════════════════════════════════════════════

    def _build_gmsh(
        self,
        r0, r1, R, L, h,
        n_r3, n_r2, n_r1, n_theta, n_z_active, n_z_deep,
        msh_path,
    ):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("DHE")

        occ = gmsh.model.occ

        # STEP 1: four rectangles in the (r, z) half-plane
        tag_r3   = occ.addRectangle(0,   0, 0, r0,    L)
        tag_r2   = occ.addRectangle(r0,  0, 0, r1-r0, L)
        tag_deep = occ.addRectangle(0,   L, 0, r1,    h-L)
        tag_ext  = occ.addRectangle(r1,  0, 0, R-r1,  h)

        all_surfs = [(2, tag_r3), (2, tag_r2), (2, tag_deep), (2, tag_ext)]

        # STEP 2: fragment — preserves shared edges
        out, _ = occ.fragment(all_surfs, [])
        occ.synchronize()
        surfs_2d = [t for d, t in out if d == 2]

        # STEP 3: revolve all surfaces 360° around z-axis at once
        occ.revolve([(2, s) for s in surfs_2d], 0, 0, 0, 0, 0, 1, 2 * math.pi)
        occ.synchronize()

        # Fragment 3D volumes for conformal interfaces
        vols_3d = gmsh.model.getEntities(3)
        if len(vols_3d) > 1:
            occ.fragment(vols_3d, [])
            occ.synchronize()

        # STEP 4: physical groups
        self._assign_physical_groups(r0, r1, R, L, h)

        # STEP 5: mesh-size refinement near pipe walls
        self._set_mesh_size_fields(r0, r1, R, L, h, n_r3, n_r2, n_r1, n_z_active)

        # STEP 6: generate and export
        gmsh.model.mesh.generate(3)
        gmsh.write(msh_path)
        gmsh.finalize()

    def _assign_physical_groups(self, r0, r1, R, L, h):
        tol = 1e-3 * min(r0, r1 - r0, R - r1, L, h - L)

        vols  = gmsh.model.getEntities(3)
        surfs = gmsh.model.getEntities(2)

        omega1, omega2, omega3 = [], [], []
        s10, s11, s12, s13, s20, s21, s22 = [], [], [], [], [], [], []

        for _, vtag in vols:
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(3, vtag)
            cr = math.hypot(cx, cy)
            if cr > r1 + tol:
                omega1.append(vtag)
            elif cr < r1 - tol and cz > L + tol:
                omega1.append(vtag)
            elif r0 + tol < cr < r1 - tol and cz < L - tol:
                omega2.append(vtag)
            elif cr < r0 - tol and cz < L - tol:
                omega3.append(vtag)

        for _, stag in surfs:
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, stag)
            cr = math.hypot(cx, cy)

            if abs(cz) < tol:
                if cr > r1 - tol:
                    s10.append(stag)
                elif cr > r0 - tol:
                    s11.append(stag)
                else:
                    s12.append(stag)
            elif abs(cz - h) < tol or abs(cr - R) < tol:
                s13.append(stag)
            elif abs(cr - r1) < tol and cz < L - tol:
                s20.append(stag)
            elif abs(cr - r0) < tol and cz < L - tol:
                s21.append(stag)
            elif abs(cz - L) < tol and cr < r1 - tol:
                s22.append(stag)

        def add_vol(tags, tag):
            if tags:
                gmsh.model.addPhysicalGroup(3, tags, tag)

        def add_surf(tags, tag):
            if tags:
                gmsh.model.addPhysicalGroup(2, tags, tag)

        add_vol(omega1, self.TAG_OMEGA1)
        add_vol(omega2, self.TAG_OMEGA2)
        add_vol(omega3, self.TAG_OMEGA3)

        add_surf(s10, self.TAG_D2_ROCK)
        add_surf(s11, self.TAG_D2_INLET)
        add_surf(s12, self.TAG_D2_OUTLET)
        add_surf(s13, self.TAG_D1_GEXT)
        add_surf(s20, self.TAG_C1)
        add_surf(s21, self.TAG_C2)
        add_surf(s22, self.TAG_CL)

    def _set_mesh_size_fields(self, r0, r1, R, L, h, n_r3, n_r2, n_r1, n_z_active):
        curves = gmsh.model.getEntities(1)
        tol = 1e-3 * min(r0, r1 - r0, R - r1, L)

        pipe_curves = []
        for _, ctag in curves:
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(1, ctag)
            cr = math.hypot(cx, cy)
            if abs(cr - r0) < tol or abs(cr - r1) < tol:
                pipe_curves.append(ctag)

        if not pipe_curves:
            return

        size_r3 = r0      / max(n_r3, 1)
        size_r2 = (r1-r0) / max(n_r2, 1)
        size_r1 = (R-r1)  / max(n_r1, 1)
        size_z  = L       / max(n_z_active, 1)
        size_min = min(size_r3, size_r2, size_z)
        size_max = size_r1

        fid_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fid_dist, "CurvesList", pipe_curves)
        gmsh.model.mesh.field.setNumber(fid_dist, "Sampling", 100)

        fid_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fid_thr, "InField", fid_dist)
        gmsh.model.mesh.field.setNumber(fid_thr, "SizeMin", size_min)
        gmsh.model.mesh.field.setNumber(fid_thr, "SizeMax", size_max)
        gmsh.model.mesh.field.setNumber(fid_thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(fid_thr, "DistMax", (R - r1) * 0.5)

        gmsh.model.mesh.field.setAsBackgroundMesh(fid_thr)

    # ════════════════════════════════════════════════════════════════
    # meshio conversion
    # ════════════════════════════════════════════════════════════════

    def _convert_meshio(self, msh_path, cells_xdmf, facets_xdmf):
        msh = meshio.read(msh_path)

        tet_cells = next(
            (b.data for b in msh.cells if b.type == "tetra"), None)
        if tet_cells is None:
            raise RuntimeError("No tetrahedra in Gmsh output.")
        tet_data = msh.cell_data_dict["gmsh:physical"]["tetra"]

        meshio.write(cells_xdmf, meshio.Mesh(
            points=msh.points,
            cells=[("tetra", tet_cells)],
            cell_data={"cell_tags": [tet_data]},
        ))

        tri_cells = next(
            (b.data for b in msh.cells if b.type == "triangle"), None)
        if tri_cells is None:
            raise RuntimeError("No triangle facets in Gmsh output.")
        tri_data = msh.cell_data_dict["gmsh:physical"]["triangle"]

        meshio.write(facets_xdmf, meshio.Mesh(
            points=msh.points,
            cells=[("triangle", tri_cells)],
            cell_data={"facet_tags": [tri_data]},
        ))

    # ════════════════════════════════════════════════════════════════
    # DOLFINx loading and measures
    # ════════════════════════════════════════════════════════════════

    def _load_dolfinx(self, cells_xdmf, facets_xdmf):
        comm = MPI.COMM_WORLD

        with XDMFFile(comm, cells_xdmf, "r") as xf:
            self.mesh = xf.read_mesh(name="Grid")
            self.cell_tags = xf.read_meshtags(self.mesh, name="Grid")

        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        self.mesh.topology.create_connectivity(fdim, tdim)
        self.mesh.topology.create_connectivity(tdim, fdim)

        with XDMFFile(comm, facets_xdmf, "r") as xf:
            self.facet_tags = xf.read_meshtags(self.mesh, name="Grid")

    def _create_measures(self):
        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags)
        self.n  = ufl.FacetNormal(self.mesh)
