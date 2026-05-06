"""
_result.py
==========

Contenedor de resultados de simulación con save/load (.npz) y export_xdmf.

Estructura del .npz
-------------------
    nodes          : (N_nodes, 3)
    elements       : (N_tets, 4)
    boundary_faces : (N_faces, 3)
    cell_tags      : (N_tets,) int
    facet_tags     : (N_faces,) int
    times          : (N_snaps,)
    T              : (N_snaps, N_nodes)
    geom_r0, geom_r1, geom_R, geom_L, geom_h : float
"""

import numpy as np


class DHEResult:
    """
    Resultado de una simulación acoplada DHE.

    Attributes
    ----------
    nodes : ndarray (N_nodes, 3)
    elements : ndarray (N_tets, 4)
    boundary_faces : ndarray (N_faces, 3)
    cell_tags : ndarray (N_tets,) int
    facet_tags : ndarray (N_faces,) int
    times : ndarray (N_snaps,)
    T : ndarray (N_snaps, N_nodes)
    geometry_params : dict — {r0, r1, R, L, h}
    """

    def __init__(self, nodes, elements, boundary_faces,
                 cell_tags, facet_tags,
                 times, T,
                 geometry_params=None):
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements)
        self.boundary_faces = np.asarray(boundary_faces)
        self.cell_tags = np.asarray(cell_tags, dtype=int)
        self.facet_tags = np.asarray(facet_tags, dtype=int)
        self.times = np.asarray(times)
        self.T = np.asarray(T)  # (N_snaps, N_nodes)
        self.geometry_params = geometry_params or {}

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save(self, path):
        """Guarda el resultado en un archivo .npz."""
        geo = self.geometry_params
        np.savez_compressed(
            path,
            nodes=self.nodes,
            elements=self.elements,
            boundary_faces=self.boundary_faces,
            cell_tags=self.cell_tags,
            facet_tags=self.facet_tags,
            times=self.times,
            T=self.T,
            geom_r0=geo.get('r0', np.nan),
            geom_r1=geo.get('r1', np.nan),
            geom_R=geo.get('R', np.nan),
            geom_L=geo.get('L', np.nan),
            geom_h=geo.get('h', np.nan),
        )

    @classmethod
    def load(cls, path):
        """Carga desde un archivo .npz."""
        data = np.load(path, allow_pickle=False)
        geo = {
            'r0': float(data['geom_r0']),
            'r1': float(data['geom_r1']),
            'R': float(data['geom_R']),
            'L': float(data['geom_L']),
            'h': float(data['geom_h']),
        }
        return cls(
            nodes=data['nodes'],
            elements=data['elements'],
            boundary_faces=data['boundary_faces'],
            cell_tags=data['cell_tags'],
            facet_tags=data['facet_tags'],
            times=data['times'],
            T=data['T'],
            geometry_params=geo,
        )

    # ------------------------------------------------------------------
    # Exportación XDMF (ParaView)
    # ------------------------------------------------------------------
    def export_xdmf(self, filename='output'):
        """
        Exporta a XDMF + HDF5 para visualización en ParaView.
        El cell_tags se añade como CellData.
        """
        import h5py

        h5_path = filename + '.h5'
        xdmf_path = filename + '.xdmf'

        N_nodes = self.nodes.shape[0]
        N_tets = self.elements.shape[0]
        N_snaps = len(self.times)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('nodes', data=self.nodes)
            f.create_dataset('elements', data=self.elements)
            f.create_dataset('cell_tags', data=self.cell_tags)
            f.create_dataset('times', data=self.times)
            f.create_dataset('T', data=self.T)

        # Escribir XDMF manualmente
        lines = ['<?xml version="1.0"?>',
                 '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
                 '<Xdmf Version="3.0">',
                 '  <Domain>',
                 '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">']

        for i, t in enumerate(self.times):
            lines += [
                f'      <Grid Name="step_{i}" GridType="Uniform">',
                f'        <Time Value="{t}"/>',
                f'        <Topology TopologyType="Tetrahedron" NumberOfElements="{N_tets}">',
                f'          <DataItem Dimensions="{N_tets} 4" NumberType="Int" Format="HDF">',
                f'            {h5_path}:/elements',
                f'          </DataItem>',
                f'        </Topology>',
                f'        <Geometry GeometryType="XYZ">',
                f'          <DataItem Dimensions="{N_nodes} 3" NumberType="Float" Format="HDF">',
                f'            {h5_path}:/nodes',
                f'          </DataItem>',
                f'        </Geometry>',
                f'        <Attribute Name="T" AttributeType="Scalar" Center="Node">',
                f'          <DataItem Dimensions="{N_nodes}" NumberType="Float" Format="HDF"',
                f'                    ItemType="HyperSlab">',
                f'            <DataItem Dimensions="3 2" Format="XML">',
                f'              {i} 0  1 1  1 {N_nodes}',
                f'            </DataItem>',
                f'            <DataItem Dimensions="{N_snaps} {N_nodes}" Format="HDF">',
                f'              {h5_path}:/T',
                f'            </DataItem>',
                f'          </DataItem>',
                f'        </Attribute>',
                f'        <Attribute Name="subdomain" AttributeType="Scalar" Center="Cell">',
                f'          <DataItem Dimensions="{N_tets}" NumberType="Int" Format="HDF">',
                f'            {h5_path}:/cell_tags',
                f'          </DataItem>',
                f'        </Attribute>',
                f'      </Grid>',
            ]

        lines += ['    </Grid>', '  </Domain>', '</Xdmf>']

        with open(xdmf_path, 'w') as f:
            f.write('\n'.join(lines))

        return xdmf_path, h5_path

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------
    def extract_subdomain(self, omega_idx):
        """
        Retorna (nodes_sub, T_sub) para el subdominio dado (1, 2 o 3).
        T_sub tiene shape (N_snaps, N_nodes_sub).
        """
        elem_mask = self.cell_tags == omega_idx
        node_idx = np.unique(self.elements[elem_mask].ravel())
        return self.nodes[node_idx], self.T[:, node_idx]

    def borehole_profile(self, snap_idx=-1):
        """
        Perfil promedio radial vs z para el snapshot dado.
        Promedia sobre θ (todos los nodos con r < r1).

        Returns
        -------
        z_vals : ndarray
        T_mean : ndarray
        """
        geo = self.geometry_params
        r1 = geo.get('r1', np.inf)
        x, y, z = self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2]
        r = np.sqrt(x**2 + y**2)
        mask = r < r1
        z_sub = z[mask]
        T_sub = self.T[snap_idx, mask]
        order = np.argsort(z_sub)
        return z_sub[order], T_sub[order]

