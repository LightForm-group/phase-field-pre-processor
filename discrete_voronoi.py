import numpy as np
from plotly import graph_objects
from plotly.colors import qualitative
from vecmaths.geometry import get_box_xyz

from scipy.spatial import KDTree


def euclidean_distance_matrix(a, b, use_loop=False):
    print(f"a.shape: {a.shape}; b.shape: {b.shape}")
    if use_loop:
        dist = []
        for b_i in b:
            dist_i = np.linalg.norm(a[:, None, :] - b_i[None, None, :], axis=-1)
            dist.append(dist_i)
        dist = np.array(dist)
        return dist
    else:
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def get_coordinate_grid(size, grid_size):
    """Get the coordinates of the element centres of a uniform grid."""

    grid_size = np.array(grid_size)
    size = np.array(size)

    grid = np.meshgrid(*[np.arange(i) for i in grid_size])
    grid = np.moveaxis(np.array(grid), 0, -1)  # shape (*grid_size, dimension)

    element_size = (size / grid_size).reshape(1, 1, -1)

    coords = grid * size.reshape(1, 1, -1) / grid_size.reshape(1, 1, -1)
    coords += element_size / 2

    return coords, element_size


class DiscreteVoronoi:
    def __init__(
        self,
        seeds,
        grid_size,
        size=None,
        periodic=False,
    ):
        """
        Parameters
        ----------
        seeds : list or ndarray of shape (N, 2) or (N, 3)
            Row vectors of seed positions in 2D or 3D
        grid_size : list or ndarray of length 2 or 3
        size : list or ndarray of length 2 or 3, optional
            If not specified, a unit square/box is used.
        periodic : bool, optional
            Should the seeds and box be considered periodic. By default, False.

        """

        seeds = np.asarray(seeds)
        grid_size = np.asarray(grid_size)
        dimension = grid_size.size

        if size is None:
            size = [1] * dimension
        size = np.asarray(size)

        self.seeds = seeds
        self.grid_size = grid_size
        self.dimension = dimension
        self.size = size
        self.periodic = periodic

        self.element_size, self.coords, self.voxel_assignment = self._assign_voxels()

        self.coords_flat = self.coords.reshape(-1, self.dimension)
        self.voxel_assignment_flat = self.voxel_assignment.reshape(-1)

    def _assign_voxels(self):
        """Assign voxels to their closest seed point.

        Returns
        -------
        tuple
            element_size : ndarray of shape (`dimension`,)
            coords: ndarray of shape (*grid_size, `dimension`)
                Cartesian coordinates of the element centres
            voxel_assignment : ndarray of shape `grid_size`
                The index of the closest seed for each voxel.

        """

        # Get coordinates of grid element centres:
        coords, element_size = get_coordinate_grid(self.size, self.grid_size)
        coords_flat = coords.reshape(-1, self.dimension)

        boxsize = self.size if self.periodic else None
        tree = (
            KDTree(self.seeds, boxsize=boxsize, balanced_tree=False)
            if self.periodic
            else KDTree(self.seeds)
        )
        voxel_assignment = tree.query(coords_flat)[1].reshape(coords.shape[:-1])

        return element_size.flatten(), coords, voxel_assignment

    @property
    def num_seeds(self):
        return self.seeds.shape[0]

    @property
    def seeds_periodic_mapping(self):
        "Map periodic seed indices back to base seed indices."
        return np.tile(np.arange(self.seeds.shape[0]), 3**self.dimension)

    @property
    def seeds_periodic(self):
        """Get seeds positions including neighbouring periodic images."""
        trans_facts_2D = np.array(
            [
                [0, 0],
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )
        if self.dimension == 2:
            trans = self.size * trans_facts_2D
        else:
            trans = self.size * np.vstack(
                [
                    np.hstack([trans_facts_2D, -np.ones((trans_facts_2D.shape[0], 1))]),
                    np.hstack([trans_facts_2D, np.zeros((trans_facts_2D.shape[0], 1))]),
                    np.hstack([trans_facts_2D, np.ones((trans_facts_2D.shape[0], 1))]),
                ]
            )

        seeds_periodic = np.concatenate(trans[:, None] + self.seeds)
        return seeds_periodic

    def show(self, show_voxels=False, show_periodic_seeds=False, layout=None):
        """Show the discrete Voronoi tessellation."""

        layout = layout or {}

        if self.dimension == 2:
            edge_vecs = np.diag(np.concatenate([self.size, [0]]))
            box_xyz = get_box_xyz(edge_vecs, faces=True)["face01a"][0][[0, 1]]
        else:
            edge_vecs = np.diag(self.size)
            box_xyz = get_box_xyz(edge_vecs)[0]

        region_boundaries_approx_idx = np.where(
            np.diff(self.voxel_assignment, axis=0) != 0
        )
        region_boundaries_approx = self.coords[region_boundaries_approx_idx]

        seed_data = self.seeds_periodic if show_periodic_seeds else self.seeds

        if self.dimension == 2:
            data = [
                {
                    "x": box_xyz[0],
                    "y": box_xyz[1],
                    "mode": "lines",
                    "name": "Box",
                },
                {
                    "x": region_boundaries_approx[:, 0],
                    "y": region_boundaries_approx[:, 1],
                    "name": "Boundaries approx.",
                    "mode": "markers",
                },
                {
                    "x": seed_data[:, 0],
                    "y": seed_data[:, 1],
                    "mode": "markers",
                    "name": "Seeds",
                    "marker": {
                        "size": 10,
                    },
                },
            ]
            if show_voxels:
                data.append(
                    {
                        "type": "scattergl",
                        "x": self.coords_flat[:, 0],
                        "y": self.coords_flat[:, 1],
                        "mode": "markers",
                        "marker": {
                            "color": self.voxel_assignment_flat,
                            "colorscale": qualitative.D3,
                            "size": 4,
                        },
                        "name": "Voxels",
                    }
                )

            layout.update(
                {
                    "xaxis": {
                        "scaleanchor": "y",
                        "showgrid": False,
                        # 'dtick': self.element_size[0],
                    },
                    "yaxis": {
                        "showgrid": False,
                        # 'dtick': self.element_size[1],
                    },
                }
            )
        else:
            data = [
                {
                    "type": "scatter3d",
                    "x": box_xyz[0],
                    "y": box_xyz[1],
                    "z": box_xyz[2],
                    "mode": "lines",
                    "name": "Box",
                },
                {
                    "type": "scatter3d",
                    "x": region_boundaries_approx[:, 0],
                    "y": region_boundaries_approx[:, 1],
                    "z": region_boundaries_approx[:, 2],
                    "name": "Boundaries approx.",
                    "mode": "markers",
                    "marker": {
                        "size": 2,
                    },
                },
                {
                    "type": "scatter3d",
                    "x": seed_data[:, 0],
                    "y": seed_data[:, 1],
                    "z": seed_data[:, 2],
                    "mode": "markers",
                    "name": "Seeds",
                    "marker": {
                        "size": 10,
                    },
                },
            ]
            if show_voxels:
                data.append(
                    {
                        "type": "scatter3d",
                        "x": self.coords_flat[:, 0],
                        "y": self.coords_flat[:, 1],
                        "z": self.coords_flat[:, 2],
                        "mode": "markers",
                        "marker": {
                            "color": self.voxel_assignment_flat,
                            "size": 4,
                        },
                        "name": "Voxels",
                    }
                )

        fig = graph_objects.FigureWidget(
            data=data,
            layout=layout,
        )

        return fig
