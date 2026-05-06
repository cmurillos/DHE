"""
_coupled_mesh.py
================

Construye la malla tetraédrica global del dominio acoplado Ω = Ω1 ∪ Ω2 ∪ Ω3.

Estrategia
----------
1. Generar nodos estructurados en (r, θ, z) que incluyan las superficies
   críticas r ∈ {r0, r1, R} y z ∈ {0, L, h} para garantizar interfaces conformes.
2. Convertir a coordenadas cartesianas.
3. Tetraedrizar con scipy.spatial.Delaunay.
4. Filtrar simplices cuyo centroide caiga fuera del dominio.
5. Etiquetar celdas y facets con DHEGeometry.
6. Construir MeshTet de scikit-fem con subdomains/facet dicts.
"""

import warnings

import numpy as np
from scipy.spatial import Delaunay
from skfem import MeshTet

from ._geometry import DHEGeometry


class CoupledCylinderMesh:
    """
    Malla tetraédrica global para el dominio acoplado.

    Attributes
    ----------
    geometry : DHEGeometry
    nodes : ndarray (N_nodes, 3)
    elements : ndarray (N_tets, 4)
    boundary_faces : ndarray (N_faces, 3)
    cell_tags : ndarray (N_tets,) int   — 1=Ω1, 2=Ω2, 3=Ω3
    facet_tags : ndarray (N_faces,) int — códigos de DHEGeometry.label_facets
    skfem_mesh : skfem.MeshTet
    cell_subdomains : dict[str, ndarray]
    facet_subdomains : dict[str, ndarray]
    """

    CELL_TAG_NAMES = {1: 'omega1', 2: 'omega2', 3: 'omega3'}

    FACET_TAG_NAMES = {
        10: 'C1',
        20: 'C2',
        31: 'C_L2',
        32: 'C_L3',
        41: 'D2_omega1',
        42: 'D2_omega2',
        43: 'D2_omega3',
        50: 'D1',
        60: 'Gamma_ext',
    }

    def __init__(
        self,
        geometry: DHEGeometry,
        nr_omega1: int = 8,
        nr_omega2: int = 4,
        nr_omega3: int = 4,
        nz_above_L: int = 20,
        nz_below_L: int = 10,
        ntheta: int = 16,
    ):
        self.geometry = geometry
        self.nr_omega1 = nr_omega1
        self.nr_omega2 = nr_omega2
        self.nr_omega3 = nr_omega3
        self.nz_above_L = nz_above_L
        self.nz_below_L = nz_below_L
        self.ntheta = ntheta

        self.generate_nodes()
        self.tetrahedralize()
        self.tag_cells()
        self.tag_facets()
        self.build_skfem_mesh()

    # ------------------------------------------------------------------
    # 1. Generación de nodos
    # ------------------------------------------------------------------
    def generate_nodes(self):
        """
        Genera nodos en cilíndricas y almacena como cartesianos en self.nodes.

        Para z ∈ [0, L]: radios r ∈ {0…r0, r0…r1, r1…R} (3 subdominios presentes).
        Para z ∈ [L, h]: radios r ∈ {0…R} (sólo Ω1, tapón inferior de subsuelo).
        Las superficies r ∈ {r0, r1, R} y z ∈ {0, L, h} están incluidas
        explícitamente para interfaces conformes.
        """
        geo = self.geometry
        r0, r1, R = geo.r0, geo.r1, geo.R
        L, h = geo.L, geo.h

        # Particiones radiales
        r_inner = np.linspace(0.0, r0, self.nr_omega3 + 1)
        r_annular = np.linspace(r0, r1, self.nr_omega2 + 1)
