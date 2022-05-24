import numpy as np
from plotly import graph_objects
from plotly.colors import qualitative
from vecmaths.geometry import get_box_xyz

from scipy.spatial import KDTree

from voxel_map import VoxelMap


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


class DiscreteVoronoi(VoxelMap):
    def __init__(
        self,
        region_seeds,
        grid_size,
        size=None,
        is_periodic=True,
        random_seed=None,
    ):
        """
        Parameters
        ----------
        region_seeds : list or ndarray of shape (N, 2) or (N, 3)
            Row vectors of seed positions in 2D or 3D. Must be coordinates within real-space
            `size`.
        grid_size : list or ndarray of length 2 or 3
        size : list or ndarray of length 2 or 3, optional
            If not specified, a unit square/box is used.
        is_periodic : bool, optional
            Should the seeds and box be considered periodic. By default, True.

        """

        region_seeds = np.asarray(region_seeds)
        grid_size = np.asarray(grid_size)
        dimension = grid_size.size

        if size is None:
            size = [1] * dimension
        size = np.asarray(size)

        if size.size != grid_size.size:
            raise ValueError(
                f"`size` ({size}) implies {size.size} dimensions, but `grid_size` "
                f"({grid_size}) implies {grid_size.size} dimensions."
            )

        if region_seeds.shape[1] != grid_size.size:
            raise ValueError(
                f"Specified `region_seeds` should be of shape (num_phases, "
                f"{grid_size.size}), but `region_seeds` has shape: {region_seeds.shape}"
            )

        bad_seeds = np.logical_or(region_seeds < 0, region_seeds > size)
        if np.any(bad_seeds):
            raise ValueError(
                f"`region_seeds` must be coordinates within `size` ({size})."
            )

        self.seeds = region_seeds
        self.seeds_grid = (grid_size * region_seeds / size).astype(int)
        self.random_seed = random_seed

        region_ID = self._get_region_ID(size, grid_size, is_periodic, dimension)

        super().__init__(
            region_ID=region_ID,
            size=size,
            is_periodic=is_periodic,
        )

    @classmethod
    def from_random(
        cls,
        size,
        grid_size,
        num_regions=None,
        random_seed=None,
        is_periodic=True,
    ):
        region_seeds = cls.get_random_seeds(
            num_regions=num_regions,
            size=size,
            random_seed=random_seed,
        )
        return cls(
            size=size,
            grid_size=grid_size,
            region_seeds=region_seeds,
            random_seed=random_seed,
            is_periodic=is_periodic,
        )

    @classmethod
    def from_seeds(
        cls,
        size,
        grid_size,
        region_seeds=None,
        random_seed=None,
        is_periodic=True,
    ):
        return cls(
            size=size,
            grid_size=grid_size,
            region_seeds=region_seeds,
            random_seed=random_seed,
            is_periodic=is_periodic,
        )

    @staticmethod
    def get_random_seeds(num_regions, size, random_seed=None):
        size = np.asarray(size)
        rng = np.random.default_rng(seed=random_seed)
        seeds = rng.random((num_regions, size.size)) * size
        return seeds

    def _get_region_ID(self, size, grid_size, is_periodic, dimension):
        """Assign voxels to their closest seed point.

        Returns
        -------
        region_ID
            The index of the closest seed for each voxel.

        """

        # Get coordinates of grid element centres:
        coords, _ = get_coordinate_grid(size, grid_size)
        coords_flat = coords.reshape(-1, dimension)
        tree = KDTree(self.seeds_grid, boxsize=size if is_periodic else None)
        region_ID = tree.query(coords_flat)[1].reshape(coords.shape[:-1])

        return region_ID

    @property
    def num_seeds(self):
        return self.seeds_grid.shape[0]

    @property
    def seeds_periodic_mapping(self):
        "Map periodic seed indices back to base seed indices."
        return np.tile(np.arange(self.seeds_grid.shape[0]), 3**self.dimension)

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

        seeds_periodic = np.concatenate(trans[:, None] + self.seeds_grid)
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

        seed_data = self.seeds_periodic if show_periodic_seeds else self.seeds_grid

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
