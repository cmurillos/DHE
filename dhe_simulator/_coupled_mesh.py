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
        r_outer = np.linspace(r1, R, self.nr_omega1 + 1)
        r_above = np.unique(np.concatenate([r_inner, r_annular, r_outer]))

        r_below = np.unique(np.concatenate([
            np.linspace(0.0, r1, self.nr_omega3 + self.nr_omega2 + 1),
            np.linspace(r1, R, self.nr_omega1 + 1),
        ]))

        # Particiones axiales
        z_above = np.linspace(0.0, L, self.nz_above_L + 1)
        z_below = np.linspace(L, h, self.nz_below_L + 1)
        z_all = np.unique(np.concatenate([z_above, z_below]))

        # Ángulos (excluir 2π para no duplicar θ=0)
        thetas = np.linspace(0, 2 * np.pi, self.ntheta + 1)[:-1]

        node_list = []

        # Nodo en r=0 por cada nivel z (evitar singularidad del eje)
        for iz, z in enumerate(z_all):
            node_list.append([0.0, 0.0, z])

        # Nodos para z ∈ [0, L] con r > 0
        for iz, z in enumerate(z_above):
            # Staggering angular alternado para evitar tetraedros degenerados
            theta_offset = (iz % 2) * (np.pi / self.ntheta)
            for r in r_above:
                if r == 0.0:
                    continue  # ya se añadió el eje arriba
                for ith, theta in enumerate(thetas):
                    t = theta + theta_offset
                    node_list.append([r * np.cos(t), r * np.sin(t), z])

        # Nodos para z ∈ (L, h] con r > 0 (sólo Ω1)
        for iz, z in enumerate(z_below[1:], start=1):  # skip z=L (ya incluido arriba)
            theta_offset = (iz % 2) * (np.pi / self.ntheta)
            for r in r_below:
                if r == 0.0:
                    continue
                for ith, theta in enumerate(thetas):
                    t = theta + theta_offset
                    node_list.append([r * np.cos(t), r * np.sin(t), z])

        self.nodes = np.array(node_list, dtype=float)

    # ------------------------------------------------------------------
    # 2. Tetraedrización
    # ------------------------------------------------------------------
    def tetrahedralize(self):
        """
        Tetraedriza los nodos con Delaunay y filtra tetraedros fuera del dominio.

        Además de las caras exteriores (boundary_faces), identifica las caras
        interiores que separan subdominios distintos (interface_faces), ya que
        las interfaces C1 y C2 son superficies cilíndricas interiores y NO
        aparecen en boundary_faces (sus caras triangulares aparecen dos veces
        en la lista global de caras, una por cada tetraedro adyacente).

        El etiquetado de interfaces se hace POR ADYACENCIA: una cara interior
        con tetraedros de distintos subdominios a ambos lados es una interfaz.
        Esto es más robusto que detectar r=r1 en los centroides de los triángulos
        (que rara vez cae exactamente en el valor exacto con Delaunay).
        """
        from collections import Counter, defaultdict

        geo = self.geometry
        tri = Delaunay(self.nodes)

        # Calcular centroides de cada simplice
        pts = self.nodes[tri.simplices]
        centroids = pts.mean(axis=1)
        cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]

        # Mantener sólo tetraedros dentro del dominio
        in_domain = (
            geo.in_omega1(cx, cy, cz) |
            geo.in_omega2(cx, cy, cz) |
            geo.in_omega3(cx, cy, cz)
        )
        self.elements = tri.simplices[in_domain]

        # Construir mapa: cara -> lista de índices de tetraedros
        face_to_tets = defaultdict(list)
        for tet_idx, tet in enumerate(self.elements):
            for i in range(4):
                face = tuple(sorted([tet[j] for j in range(4) if j != i]))
                face_to_tets[face].append(tet_idx)

        boundary_list = []
        interface_list = []  # (cara, tet_izq, tet_der)
        for face, tet_ids in face_to_tets.items():
            if len(tet_ids) == 1:
                boundary_list.append(list(face))
            elif len(tet_ids) == 2:
                interface_list.append((list(face), tet_ids[0], tet_ids[1]))

        self.boundary_faces = np.array(boundary_list, dtype=int) if boundary_list else np.empty((0, 3), dtype=int)
        # Guardar interfaces para etiquetado posterior
        self._interface_faces = interface_list  # lista de (cara3nodos, tet0, tet1)

    # ------------------------------------------------------------------
    # 3. Etiquetado de celdas
    # ------------------------------------------------------------------
    def tag_cells(self):
        """Calcula centroides de cada tetraedro y etiqueta con label_cells."""
        pts = self.nodes[self.elements]
        centroids = pts.mean(axis=1)
        self.cell_tags = self.geometry.label_cells(centroids)

    # ------------------------------------------------------------------
    # 4. Etiquetado de facets de frontera e interfaces internas
    # ------------------------------------------------------------------
    def tag_facets(self):
        """
        Etiqueta las caras exteriores (boundary_faces) con label_facets,
        y detecta las caras interiores de interfaz por adyacencia de subdominios.

        Interfaces detectadas por par de subdominios adyacentes:
            (Ω1, Ω2) → C1  (tag=10)
            (Ω2, Ω3) → C2  (tag=20)
            (Ω1, Ω2) con z~L → C_L2 (tag=31)
            (Ω1, Ω3) con z~L → C_L3 (tag=32)

        Para distinguir C1 (lateral) de C_L2 (fondo) cuando ambas son (Ω1, Ω2),
        usamos la posición z del centroide de la cara.
        """
        geo = self.geometry

        # Caras exteriores
        if len(self.boundary_faces) > 0:
            ext_centroids = self.nodes[self.boundary_faces].mean(axis=1)
            self.facet_tags = geo.label_facets(ext_centroids)
        else:
            self.facet_tags = np.array([], dtype=int)

        # Caras interiores de interfaz: detectadas por adyacencia de cell_tags
        iface_faces = []
        iface_tags = []
        tol = geo.tol

        for face_nodes, ti, tj in self._interface_faces:
            tag_i = self.cell_tags[ti]
            tag_j = self.cell_tags[tj]
            pair = tuple(sorted([tag_i, tag_j]))

            if pair == (1, 2):
                # C1 (lateral) o C_L2 (fondo): distinguir por z del centroide
                centroid = self.nodes[face_nodes].mean(axis=0)
                if abs(centroid[2] - geo.L) < geo.L * 0.05:
                    # z ≈ L → fondo del anular C_L2
                    tag = 31
                else:
                    # Pared lateral C1
                    tag = 10
                iface_faces.append(face_nodes)
                iface_tags.append(tag)

            elif pair == (1, 3):
                # C_L3 (fondo del tubo interior): siempre en z=L
                iface_faces.append(face_nodes)
                iface_tags.append(32)

            elif pair == (2, 3):
                # C2 (pared interior tubería)
                iface_faces.append(face_nodes)
                iface_tags.append(20)

        if iface_faces:
            self.interface_faces = np.array(iface_faces, dtype=int)
            self.interface_tags = np.array(iface_tags, dtype=int)
        else:
            self.interface_faces = np.empty((0, 3), dtype=int)
            self.interface_tags = np.array([], dtype=int)

    # ------------------------------------------------------------------
    # 5. Conversión a scikit-fem
    # ------------------------------------------------------------------
    def build_skfem_mesh(self):
        """
        Crea self.skfem_mesh = MeshTet con subdomains y boundaries marcados.
        scikit-fem espera (3, N_nodes) y (4, N_tets) — matrices transpuestas.

        Para poder usar FacetBasis con interfaces internas (C1, C2, C_L),
        scikit-fem necesita que esas caras estén registradas como facets del mesh.
        Dado que scikit-fem incluye TODOS los facets (internos y externos) en
        mesh.facets, construimos el mapa desde la lista global de facets.
        """
        p = self.nodes.T       # (3, N_nodes)
        t = self.elements.T    # (4, N_tets)

        self.skfem_mesh = MeshTet(p, t, validate=False)

        # Diccionarios subdominio -> índices de celdas
        self.cell_subdomains = {}
        for tag, name in self.CELL_TAG_NAMES.items():
            idx = np.where(self.cell_tags == tag)[0]
            if len(idx) > 0:
                self.cell_subdomains[name] = idx

        # Mapa: frozenset de nodos → índice de facet en mesh.facets (incluye internos)
        skfem_facets = self.skfem_mesh.facets.T  # (N_all_facets, 3)
        facet_to_skfem = {}
        for i, f in enumerate(skfem_facets):
            facet_to_skfem[frozenset(f)] = i

        self.facet_subdomains = {}

        # Caras exteriores (boundary_faces)
        for tag, name in self.FACET_TAG_NAMES.items():
            local_idx = np.where(self.facet_tags == tag)[0]
            if len(local_idx) == 0:
                continue
            skfem_idx = []
            for li in local_idx:
                key = frozenset(self.boundary_faces[li])
                if key in facet_to_skfem:
                    skfem_idx.append(facet_to_skfem[key])
            if skfem_idx:
                self.facet_subdomains[name] = np.array(skfem_idx, dtype=int)

        # Caras interiores de interfaz (C1, C2, C_L2, C_L3)
        iface_name_map = {10: 'C1', 20: 'C2', 31: 'C_L2', 32: 'C_L3'}
        iface_collections = {name: [] for name in iface_name_map.values()}

        for fi in range(len(self.interface_faces)):
            tag = self.interface_tags[fi]
            name = iface_name_map.get(tag)
            if name is None:
                continue
            key = frozenset(self.interface_faces[fi])
            if key in facet_to_skfem:
                iface_collections[name].append(facet_to_skfem[key])

        for name, idx_list in iface_collections.items():
            if idx_list:
                self.facet_subdomains[name] = np.array(idx_list, dtype=int)

    # ------------------------------------------------------------------
    # 6. Identificación de DOFs de igualdad en C_L
    # ------------------------------------------------------------------
    def get_C_L_coupling_dofs(self, basis1, basis2, basis3):
        """
        Identifica DOFs nodales en C_L para cada base de subdominio.
        Con malla conforme, los nodos en z=L son compartidos — basta buscar
        por coordenadas.

        Returns
        -------
        dofs_CL2_b1, dofs_CL2_b2, dofs_CL3_b1, dofs_CL3_b3 : ndarray
        """
        geo = self.geometry

        def dofs_on(basis, predicate_fn):
            coords = basis.doflocs  # (3, N_dofs)
            x, y, z = coords[0], coords[1], coords[2]
            mask = predicate_fn(x, y, z)
            return np.where(mask)[0]

        dofs_CL2_b1 = dofs_on(basis1, geo.on_C_L2)
        dofs_CL2_b2 = dofs_on(basis2, geo.on_C_L2)
        dofs_CL3_b1 = dofs_on(basis1, geo.on_C_L3)
        dofs_CL3_b3 = dofs_on(basis3, geo.on_C_L3)

        return dofs_CL2_b1, dofs_CL2_b2, dofs_CL3_b1, dofs_CL3_b3

    # ------------------------------------------------------------------
    # 7. Visualización
    # ------------------------------------------------------------------
    def plot(self, color_by='subdomain', show=True):
        """
        Plot 3D de la malla coloreando por subdominio o por etiqueta de frontera.
        Muestra una rodaja (slice) en θ ≈ 0 para legibilidad.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if color_by == 'subdomain':
            tags = self.cell_tags
            cmap_data = {1: 'blue', 2: 'green', 3: 'red'}
            for tag, color in cmap_data.items():
                mask = tags == tag
                if not np.any(mask):
                    continue
                centroids = self.nodes[self.elements[mask]].mean(axis=1)
                ax.scatter(
                    centroids[:, 0], centroids[:, 1], centroids[:, 2],
                    c=color, s=1, alpha=0.3, label=f'Ω{tag}'
                )
            ax.legend()
            ax.set_title('Malla por subdominio')
        elif color_by == 'facet':
            tags = self.facet_tags
            for tag, name in self.FACET_TAG_NAMES.items():
                mask = tags == tag
                if not np.any(mask):
                    continue
                centroids = self.nodes[self.boundary_faces[mask]].mean(axis=1)
                ax.scatter(
                    centroids[:, 0], centroids[:, 1], centroids[:, 2],
                    s=2, alpha=0.5, label=f'{name} ({tag})'
                )
            ax.legend(fontsize=7)
            ax.set_title('Malla por frontera')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if show:
            plt.show()
        return fig, ax
