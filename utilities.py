import json
from pathlib import Path
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from plotly import graph_objects
from plotly.colors import qualitative
from vecmaths.geometry import get_box_xyz


def euclidean_distance_matrix(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def in_hull(points, hull_points):
    """Test if a set of points are within a convex hull.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Row vectors of points to test.
    hull_points : ndarray of shape (M, 3)
        Row vectors of vertices that are used to compute a convex hull

    """

    hull = Delaunay(hull_points)
    is_in = hull.find_simplex(points) >= 0

    return is_in


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
    def __init__(self, seeds, grid_size, size=None, periodic=False, use_scipy=False):
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

        self.element_size, self.coords, self.voxel_assignment = self._assign_voxels(
            use_scipy
        )

        self.coords_flat = self.coords.reshape(-1, self.dimension)
        self.voxel_assignment_flat = self.voxel_assignment.reshape(-1)

    def _assign_voxels(self, use_scipy):
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

        # Get Euclidean distance of each grid element to each seed point
        coords_flat = coords.reshape(-1, self.dimension)

        seed_data = self.seeds_periodic if self.periodic else self.seeds

        if use_scipy:

            # Perform Voronoi tessellation of (periodic) seed positions:
            scipy_vor = Voronoi(seed_data)

            # Find which voxels are associated with each seed point:
            voxel_assignment = np.zeros(self.grid_size, dtype=int)
            for seed_idx, region_idx in enumerate(scipy_vor.point_region):
                region_verts_idx = scipy_vor.regions[region_idx]
                if region_verts_idx:
                    verts = scipy_vor.vertices[region_verts_idx]
                    voxels_in_flat = in_hull(coords_flat, verts)
                    voxels_in = voxels_in_flat.reshape(coords.shape[:-1])
                    voxels_in_idx = np.where(voxels_in)
                    voxel_assignment[voxels_in_idx] = seed_idx
                    # print(f'voxel assignment now: \n{voxel_assignment}')
        else:

            # shape (num coords, num seeds)
            dist = euclidean_distance_matrix(coords_flat, seed_data)  # BIG!

            # Assign each grid element to the nearest seed point
            nearest_seed_idx = np.argmin(dist, axis=1)  # shape (num coords,)
            voxel_assignment = nearest_seed_idx.reshape(coords.shape[:-1])

        if self.periodic:
            # Map periodic seed indices back to base seed indices:
            voxel_assignment = self.seeds_periodic_mapping[voxel_assignment]

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

    def show(self, show_voxels=False, show_periodic_seeds=False):
        """Show the discrete Voronoi tessellation."""

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
        print(region_boundaries_approx.shape)

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

            layout = {
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
            layout = {}

        fig = graph_objects.FigureWidget(
            data=data,
            layout=layout,
        )

        return fig


def write_MTEX_EBSD_file(coords, euler_angles, phases, filename):

    dimension = coords.shape[1]

    col_names = [
        "Phase",
        "x",
        "y",
        "z",
        "Euler1",
        "Euler2",
        "Euler3",
    ]
    if dimension == 2:
        col_names.pop(3)

    all_dat = np.hstack(
        [
            phases[:, None],
            coords,
            euler_angles,
        ]
    )
    header = ", ".join([f"{i}" for i in col_names])
    np.savetxt(
        fname=filename,
        header=header,
        X=all_dat,
        fmt=["%d"] + ["%20.17f"] * (len(col_names) - 1),
    )


def write_MTEX_JSON_file(data, filename):
    with Path(filename).open("w") as fh:
        json.dump(data, fh)


def unjsonify_dict(dct):
    converted_to_list = dct.pop("_converted_to_list")
    for k, v in dct.items():
        if k in converted_to_list:
            v = np.array(v)
            dct[k] = v
    return dct


def jsonify_dict(dct):
    converted_to_list = []
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
            converted_to_list.append(k)
        dct[k] = v
    dct["_converted_to_list"] = converted_to_list
    return dct


def get_by_path(root, path):
    """Get a nested dict or list item according to its "key path"

    Parameters
    ----------
    root : dict or list
        Can be arbitrarily nested.
    path : list of str
        The address of the item to get within the `root` structure.

    Returns
    -------
    sub_data : any

    """

    sub_data = root
    for key in path:
        sub_data = sub_data[key]

    return sub_data


def set_by_path(root, path, value):
    """Set a nested dict or list item according to its "key path"

    Parmaeters
    ----------
    root : dict or list
        Can be arbitrarily nested.
    path : list of str
        The address of the item to set within the `root` structure.
    value : any
        The value to set.

    """

    sub_data = root
    for key_idx, key in enumerate(path[:-1], start=1):
        try:
            sub_data = sub_data[key]
        except KeyError:
            set_by_path(sub_data, (key,), {})
            sub_data = sub_data[key]

    sub_data[path[-1]] = value


def read_shockley(theta, E_max, theta_max, degrees=True):
    """Misorientation-grain-boundary-energy relationship for low angle GBs."""

    if degrees:
        theta = np.deg2rad(theta)
        theta_max = np.deg2rad(theta_max)

    zero_idx = np.isclose(theta, 0)

    A_0 = 1 + np.log(theta_max)
    E_0 = E_max / theta_max
    E = np.zeros_like(theta)
    E[~zero_idx] = E_0 * theta[~zero_idx] * (A_0 - np.log(theta[~zero_idx]))
    E[theta > theta_max] = np.max(E[theta < theta_max])

    return E
