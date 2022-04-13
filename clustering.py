import copy
import json
from operator import le
import pickle
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations_with_replacement, combinations
from logging import getLogger, basicConfig

import numpy as np
from matflow import load_workflow
from numpy.lib.shape_base import split
import plotly
import plotly.colors
from scipy.ndimage import zoom
from plotly import graph_objects
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from utilities import (
    get_coordinate_grid,
    jsonify_dict,
    unjsonify_dict,
    write_MTEX_EBSD_file,
    write_MTEX_JSON_file,
)
from field_viz import get_volumetric_slice, RVEFieldViz, get_plotly_discrete_colour_bar
from quats import quat2euler


logger = getLogger(__name__)
basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclass
class SliceSelection:

    increment_idx: int
    is_periodic: bool
    eye: str
    up: str
    x: int = None
    y: int = None
    z: int = None


def run_matlab_script(script_name, args, nargout, script_dir=None):
    """Must be a function file (i.e. a script with a single top level function defined)."""
    import matlab.engine

    eng = matlab.engine.start_matlab()
    if script_dir:
        eng.cd(script_dir)
    out = getattr(eng, script_name)(*args, nargout=nargout)
    eng.quit()
    return out


def tile_coordinates(coords):
    """Tile 2D coordinate grid."""

    tile_X, tile_Y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    tile_disp = np.concatenate([tile_X[:, :, None], tile_Y[:, :, None]], axis=2)
    tile_disp_broadcast = tile_disp.reshape((3, 3, 1, 1, 2))
    coords_broadcast = coords.reshape(1, 1, coords.shape[0], coords.shape[1], 2)

    coords_tiled = coords_broadcast + tile_disp_broadcast

    coords_tiled_rs = np.concatenate(
        np.concatenate(
            coords_tiled,
            axis=1,
        ).swapaxes(1, 2),
        axis=0,
    ).swapaxes(0, 1)

    return coords_tiled_rs


def tile_seed_points(seed_points, grid_size):
    """
    Parameters
    ----------
    seed_points : ndarray of shape (N, 2)
    grid_size : list or ndarray of length (2,)

    Returns
    -------
    two-tuple of:
        tiled_seeds : ndarray of shape (9N, 2)
        tiling_idx_lookup : ndarray of shape (9N,)

    """

    X, Y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    tiling_arr = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(
        9, 2
    ) * np.array(grid_size)

    tiled_seeds = np.concatenate(tiling_arr[:, None] + seed_points[None])
    tiling_idx_lookup = np.tile(np.arange(seed_points.shape[0]), 9)

    return tiled_seeds, tiling_idx_lookup


def periodically_tessellate(seeds, grid_size, grid_coords=None):
    """
    Parameters
    ----------
    seeds : ndarray of shape (N, 2)
    grid_size : list of length 2
    grid_coords, ndarray of shape (M, 2), optional
        If not specified, tessellate all grid coordinates defined on the grid of
        `grid_size`.

    """

    if grid_coords is None:
        grid_coords_X, grid_coords_Y = np.meshgrid(
            np.arange(0, grid_size[0]),
            np.arange(0, grid_size[1]),
        )
        grid_coords = np.concatenate(
            [
                grid_coords_X[None],
                grid_coords_Y[None],
            ]
        ).reshape(2, -1)

    seeds_tiled, tile_index = tile_seed_points(seeds, grid_size)

    tessellated_coords = []
    tessellated_sub_grain_IDs = []
    for grid_coord in grid_coords.T:

        # Find distance to all tiled seed points:
        dist = np.linalg.norm(seeds_tiled - grid_coord[None], axis=1)
        closest_periodic_seed_idx = np.argmin(dist)

        # Find original index of this seed:
        closest_seed_idx = tile_index[closest_periodic_seed_idx]

        tessellated_coords.append(grid_coord)
        tessellated_sub_grain_IDs.append(closest_seed_idx)

    tessellated_coords = np.array(tessellated_coords)
    tessellated_sub_grain_IDs = np.array(tessellated_sub_grain_IDs)

    return tessellated_coords, tessellated_sub_grain_IDs


def remap_periodic_boundary_IDs(im, IDs_A, IDs_B):

    grain_ID_replacement_map = {}
    for idx, old_grain_ID in enumerate(IDs_A):
        new_grain_ID = IDs_B[idx]
        if old_grain_ID not in grain_ID_replacement_map:
            grain_ID_replacement_map.update({old_grain_ID: [new_grain_ID]})
        else:
            grain_ID_replacement_map[old_grain_ID].append(new_grain_ID)

    # Count how many pixels should be changed to each new ID:
    grain_ID_replacement_map_counts = {
        k: {k2: v2 for k2, v2 in zip(*np.unique(v, return_counts=True))}
        for k, v in grain_ID_replacement_map.items()
    }

    # Use the pixel majority to decide the final mapping:
    grain_ID_replacement_map_final = {}
    for k, v in grain_ID_replacement_map_counts.items():
        new_ID_trial_max_count = 0
        new_ID_trial = None
        for new_ID_i, count_k in v.items():
            if count_k > new_ID_trial_max_count:
                new_ID_trial = new_ID_i
                new_ID_trial_max_count = count_k
        grain_ID_replacement_map_final.update({k: new_ID_trial})

    # Remove trivial equal replacements:
    grain_ID_replacement_map_final = {
        k: v for k, v in grain_ID_replacement_map_final.items() if k != v
    }

    if grain_ID_replacement_map_final:
        grain_ID_replace_old, grain_ID_replace_new = zip(
            *grain_ID_replacement_map_final.items()
        )

        im = im.copy()
        for replace_idx, old_ID in enumerate(grain_ID_replace_old):
            im[im == old_ID] = grain_ID_replace_new[replace_idx]

    return im


def get_interface_map(x, split_interface=None):
    """Get a symmetric 2D interface map between unique combinations of integers
    in `x`.

    Parameters
    ----------
    x : ndarray of shape (N,)
    split_interface : ndarray of shape (N, N) of bool
        If given, assume two types of same-phase interface exist (e.g. low-angle vs
        high-angle grain boundary). The interface indices that correspond to the second
        of these types are given as `True` values in this array.

    Returns
    -------
    interfaces : ndarray of shape (M, 2)
    interface_map : ndarray of shape (N, N)

    """

    uniq_x = np.unique(x)

    num_uniq = len(uniq_x)
    num_interfaces = int(num_uniq * (num_uniq + 1) / 2)  # triangle number

    interfaces = np.empty((num_interfaces, 2), dtype=int)
    for pair_idx, pair in enumerate(combinations_with_replacement(uniq_x, r=2)):
        interfaces[pair_idx] = pair

    lookup = np.zeros((num_uniq, num_uniq), dtype=int)
    lookup_range = np.arange(num_interfaces)

    lookup[np.triu_indices(lookup.shape[0])] = lookup_range
    lookup = lookup + lookup.T - np.diag(np.diag(lookup))

    x_tile = np.tile(x, (len(x), 1))
    row_idx = x_tile.flatten()
    col_idx = x_tile.T.flatten()

    if split_interface is not None:

        lookup_split = np.ones((num_uniq * 2, num_uniq * 2), dtype=int) * -1
        lookup_split[:num_uniq, :num_uniq] = lookup
        lookup_split[
            np.arange(num_uniq, 2 * num_uniq),
            np.arange(num_uniq, 2 * num_uniq),
        ] = np.arange(num_interfaces, num_interfaces + num_uniq)

        split_interface_flat = split_interface.flatten()
        row_idx[split_interface_flat == 1] += num_uniq
        col_idx[split_interface_flat == 1] += num_uniq

        lookup = lookup_split

        interfaces = np.vstack((interfaces, np.tile(np.arange(num_uniq), (2, 1)).T))

    interface_map = lookup[row_idx, col_idx].reshape(x_tile.shape)

    return (
        interfaces,
        interface_map,
    )


def compress_1D_array(arr):

    vals = []
    nums = []
    for idx, i in enumerate(arr):

        if idx == 0:
            vals.append(i)
            nums.append(1)
            continue

        if i == vals[-1]:
            nums[-1] += 1
        else:
            vals.append(i)
            nums.append(1)

    assert sum(nums) == arr.size

    return nums, vals


def get_compressed_1D_array_string(arr, item_delim="\n"):
    out = []
    for n, v in zip(*compress_1D_array(arr)):
        out.append(f"{n} of {v}" if n > 1 else f"{v}")

    return item_delim.join(out)


def partition_sub_grain_seeds_into_grains(sub_grain_seeds, grain_IDs):

    sub_grain_seeds_idx = []
    extra_seeds = []
    grain_coords = []

    uniq_grain_IDs = np.unique(grain_IDs)

    for grain_ID_i in uniq_grain_IDs:

        sub_grain_seeds_idx_i = []

        coords_i = np.array(np.where(grain_IDs == grain_ID_i))
        grain_coords.append(coords_i)

        # Get only seeds within this grain:
        for seed_j_idx, seed_j in enumerate(sub_grain_seeds):
            if np.any(np.all(coords_i == seed_j.reshape(2, 1), axis=0)):
                sub_grain_seeds_idx_i.append(seed_j_idx)

        if not sub_grain_seeds_idx_i:
            # Add a single seed point at the first coordinate, for complete tessellation:
            extra_seeds.append(coords_i[:, 0])
            extra_seed_idx = len(sub_grain_seeds) + len(extra_seeds) - 1
            sub_grain_seeds_idx_i.append(extra_seed_idx)

        sub_grain_seeds_idx.append(sub_grain_seeds_idx_i)

    new_seeds = []
    new_sub_grain_seeds_idx = []
    is_dummy_seed = np.zeros(sub_grain_seeds.shape[0], dtype=int)
    if extra_seeds:
        sub_grain_seeds = np.vstack((sub_grain_seeds, extra_seeds))
        is_dummy_seed = np.hstack((is_dummy_seed, np.ones(len(extra_seeds), dtype=int)))

    # Reorder seeds and add on extra seeds:
    new_is_dummy_seed = []
    for sub_grain_seeds_idx_i in sub_grain_seeds_idx:
        seeds_i = sub_grain_seeds[sub_grain_seeds_idx_i]
        is_dummy_seed_i = is_dummy_seed[sub_grain_seeds_idx_i]
        new_sub_grain_seeds_idx.append(
            np.arange(len(new_seeds), len(new_seeds) + len(seeds_i))
        )
        new_seeds.extend(seeds_i)
        new_is_dummy_seed.extend(is_dummy_seed_i)

    new_seeds = np.array(new_seeds)
    new_is_dummy_seed = np.array(new_is_dummy_seed, dtype=bool)

    return new_seeds, new_sub_grain_seeds_idx, grain_coords, new_is_dummy_seed


def tessellate_sub_grain_seeds(
    sub_grain_seeds,
    sub_grain_seeds_idx,
    grid_size,
    grain_coords,
    include_grain_idx=None,
):

    sub_grain_IDs = np.ones(grid_size, dtype=int) * -1

    for idx, sub_grain_seeds_idx_i in enumerate(sub_grain_seeds_idx):

        if include_grain_idx is not None and idx not in include_grain_idx:
            continue

        tess_coords, tess_sub_grain_IDs = periodically_tessellate(
            seeds=sub_grain_seeds[sub_grain_seeds_idx_i],
            grid_size=grid_size,
            grid_coords=grain_coords[idx],
        )
        sub_grain_IDs[tess_coords[:, 0], tess_coords[:, 1]] = sub_grain_seeds_idx_i[
            tess_sub_grain_IDs
        ]

    return sub_grain_IDs


class PhaseFieldModelPreProcessor:
    """Class to perform steps that prepare a phase field model geometry from a crystal-plasticity-deformed RVE."""

    def __init__(self, workflow_dir, segmentation_sub_dir="segmentation"):
        self.workflow_dir = Path(workflow_dir)
        self.segmentation_sub_dir = self.workflow_dir.joinpath(segmentation_sub_dir)
        self.segmentation_sub_dir.mkdir(exist_ok=True)

        self.workflow = load_workflow(self.workflow_dir)

    @property
    def simulate_task(self):
        return self.workflow.tasks.simulate_volume_element_loading

    @property
    def segmentation_dirs(self):
        return list(self.workflow_dir.joinpath(self.segmentation_sub_dir).glob("*"))

    @staticmethod
    def format_segmentation_directory_name(
        element_idx, method, slice_selection, **method_kwargs
    ):
        slice_coords = {}
        if slice_selection.x is not None:
            slice_coords.update({"x": slice_selection.x})
        if slice_selection.y is not None:
            slice_coords.update({"y": slice_selection.y})
        if slice_selection.z is not None:
            slice_coords.update({"z": slice_selection.z})

        slice_key = list(slice_coords.keys())[0]
        slice_str = f"{slice_key}_{slice_coords[slice_key]}"

        per_str = "periodic" if slice_selection.is_periodic else "non-periodic"
        parametrised_path = (
            f"element_{element_idx}__inc_{slice_selection.increment_idx}__"
            f"slice_{slice_str}__{per_str}__method_{method}__"
            f"eye_{slice_selection.eye}__up_{slice_selection.up}"
        )
        if method in ["MTEX-FMC", "MTEX-FMC-new"]:
            C_Maha = method_kwargs["C_Maha"]
            smoothing = method_kwargs["smoothing"]
            parametrised_path += f"__C_Maha_{C_Maha:.2f}__smoothing_{smoothing}"

        return parametrised_path

    def get_clusterer(self, element_idx, method, slice_selection, parameters):

        clusterer_method_map = {
            "MTEX-FMC": MTEX_FMC_Clusterer,
        }

        clusterer = clusterer_method_map[method](
            pre_processor=self,
            element_idx=element_idx,
            slice_selection=slice_selection,
            method=method,
            parameters=parameters,
        )
        return clusterer


class Clusterer:
    """Class to represent the parametrised clustering process of a deformed RVE.

    Attributes
    ----------

    data : dict
        Any useful method-specific data that is generated during the clustering.

    """

    def __init__(
        self,
        pre_processor,
        element_idx,
        method,
        slice_selection,
        parameters,
    ):

        if not isinstance(slice_selection, SliceSelection):
            slice_selection = SliceSelection(**slice_selection)

        self.pre_processor = pre_processor

        self.element_idx = element_idx
        self.slice_selection = slice_selection
        self.method = method
        self.parameters = parameters

        self.seg_dir = self._get_seg_dir()

        # set in `prepare_segmentation`:
        self.slice_data = None

        # set in `do_segmentation`:
        self.grain_IDs = None
        self.grain_IDs_periodic_image = None  # set if periodic

        # set in `set_seed_points`:
        self.seed_points = None
        self.seed_points_args = None

        # set in `tessellate_seed_points`:
        self.tessellated_sub_grain_IDs = None
        self.is_dummy_seed = None
        self.sub_grain_seeds_idx = None
        self.is_interface_low_angle_GB = None

    @property
    def element(self):
        return self.pre_processor.simulate_task.elements[self.element_idx]

    @property
    def grid_size(self):
        grid_size = self.element.get_parameter_dependency_value("grid_size")
        # could have multiple grid size outputs (?):
        if isinstance(grid_size, list) and not isinstance(grid_size[0], int):
            grid_size = [i for i in grid_size if isinstance(i, list)][-1]
        return grid_size

    @property
    def size(self):
        # TODO: currently size is not a workflow parameter, so hard code this for now!
        # return self.element.get_parameter_dependency_value('size')
        return [1, 1, 1]

    @property
    def slice_grid_size(self):
        grid_size = []
        if self.slice_selection.x is None:
            grid_size.append(self.grid_size[0])
        if self.slice_selection.y is None:
            grid_size.append(self.grid_size[1])
        if self.slice_selection.z is None:
            grid_size.append(self.grid_size[2])
        return grid_size

    @property
    def slice_size(self):
        size = []
        if self.slice_selection.x is None:
            size.append(self.size[0])
        if self.slice_selection.y is None:
            size.append(self.size[1])
        if self.slice_selection.z is None:
            size.append(self.size[2])
        return size

    def format_segmentations_directory_name(self):
        slice_coords = {}
        if self.slice_selection.x is not None:
            slice_coords.update({"x": self.slice_selection.x})
        if self.slice_selection.y is not None:
            slice_coords.update({"y": self.slice_selection.y})
        if self.slice_selection.z is not None:
            slice_coords.update({"z": self.slice_selection.z})

        slice_key = list(slice_coords.keys())[0]
        slice_str = f"{slice_key}_{slice_coords[slice_key]}"

        per_str = "periodic" if self.slice_selection.is_periodic else "non-periodic"
        parametrised_path = (
            f"element_{self.element_idx}__inc_{self.slice_selection.increment_idx}__"
            f"slice_{slice_str}__{per_str}__method_{self.method}__"
            f"eye_{self.slice_selection.eye}__up_{self.slice_selection.up}"
        )
        return parametrised_path

    def _get_slice_data(self):

        coords, elem_size = get_coordinate_grid(self.slice_size, self.slice_grid_size)

        field_data = self.element.outputs.volume_element_response["field_data"]
        ori_data = field_data["O"]["data"]["quaternions"]
        phase = field_data["phase"]["data"]

        ori = ori_data[self.slice_selection.increment_idx]

        ori_slice = get_volumetric_slice(
            ori,
            x=self.slice_selection.x,
            y=self.slice_selection.y,
            z=self.slice_selection.z,
            eye=self.slice_selection.eye,
            up=self.slice_selection.up,
        )["slice_data"]
        phase_slice = get_volumetric_slice(
            phase,
            x=self.slice_selection.x,
            y=self.slice_selection.y,
            z=self.slice_selection.z,
            eye=self.slice_selection.eye,
            up=self.slice_selection.up,
        )["slice_data"]

        # print(f'phase_slice.shape: {phase_slice.shape}')

        if self.slice_selection.is_periodic:
            ori_slice = np.tile(ori_slice, (3, 3, 1))
            phase_slice = np.tile(phase_slice, (3, 3))
            coords = tile_coordinates(coords)

        coords_flat = coords.reshape(-1, 2)
        phase_flat = phase_slice.reshape(-1)
        ori_flat = ori_slice.reshape(-1, 4)

        out = {
            "coords_flat": coords_flat,
            "quats_flat": ori_flat,
            "phases_flat": phase_flat,
            "phase_names": field_data["phase"]["meta"]["phase_names"],
        }

        return out

    def get_field_data(self, data_name, data_component=None):

        field_viz = RVEFieldViz(
            volume_element_response=self.element.outputs.volume_element_response,
            increment=self.slice_selection.increment_idx,
            x=self.slice_selection.x,
            y=self.slice_selection.y,
            z=self.slice_selection.z,
            eye=self.slice_selection.eye,
            up=self.slice_selection.up,
            data_name=data_name,
            data_component=data_component,
        )
        return field_viz.plot_data

    def show_field_data(self, data_name, data_component=None):

        field_viz = RVEFieldViz(
            volume_element_response=self.element.outputs.volume_element_response,
            increment=self.slice_selection.increment_idx,
            x=self.slice_selection.x,
            y=self.slice_selection.y,
            z=self.slice_selection.z,
            eye=self.slice_selection.eye,
            up=self.slice_selection.up,
            data_name=data_name,
            data_component=data_component,
        )

        if data_name == "phase":
            colour_bar_info = get_plotly_discrete_colour_bar(
                field_viz.data_meta["phase_names"],
                plotly.colors.qualitative.D3,
            )
            colour_bar_info.update(
                {
                    "hovertext": np.array(field_viz.data_meta["phase_names"])[
                        field_viz.plot_data
                    ]
                }
            )
        else:
            colour_bar_info = {
                "colorscale": "Viridis",
                "colorbar": {},
            }
        colour_bar_info["colorbar"].update({"title": field_viz.title})

        data = [
            {
                "type": "heatmap",
                "z": field_viz.plot_data,
                "zmin": field_viz.min_value_inc_slice,
                "zmax": field_viz.max_value_inc_slice,
                **colour_bar_info,
            },
        ]
        layout = {
            "template": "none",
            # 'paper_bgcolor': 'pink',
            # 'plot_bgcolor': 'green',
            "margin": {
                "t": 50,
                "r": 0,
                "b": 50,
                "l": 0,
            },
            "height": 500,
            "width": 700,
            "showlegend": False,
            "xaxis": {
                "scaleanchor": "y",
                "constrain": "domain",
                "title": {"text": field_viz.xlabel, "font": {"size": 22}},
                "showgrid": False,
                "showticklabels": True,
                "tickmode": "array",
                "tickvals": [0, field_viz.plot_data.shape[1] - 1],
                "ticktext": [0, field_viz.plot_data.shape[1] - 1],
            },
            "yaxis": {
                "title": {"text": field_viz.ylabel, "font": {"size": 22}},
                "showgrid": False,
                "showticklabels": True,
                "tickmode": "array",
                "tickvals": [0, field_viz.plot_data.shape[0] - 1],
                "ticktext": [0, field_viz.plot_data.shape[0] - 1],
                "autorange": "reversed",  # so top-left is origin (like plt.imshow)
            },
        }
        fig = graph_objects.FigureWidget(data=data, layout=layout)

        return fig

    @property
    def JSON_path(self):
        return self.seg_dir.joinpath("clusterer.json")

    @property
    def pickle_path(self):
        return self.seg_dir.joinpath("clusterer.pkl")

    @property
    def is_segmented(self):
        if self.seg_dir.is_dir():
            return True
        else:
            return False

    @property
    def num_clustered_grains(self):
        if not self.is_segmented:
            raise ValueError("Not segmented yet; run `do_segmentation` first!")
        return len(np.unique(self.grain_IDs))

    def load(self, fmt="pickle"):
        """Load clusterer data from file."""

        path = {"json": self.JSON_path, "pickle": self.pickle_path}[fmt]
        logger.info(f"Loading {self.__class__.__name__!r} from file {str(path)}...")

        if not self.is_segmented:
            raise ValueError("Not segmented yet; run `do_segmentation` first!")

        if fmt == "pickle":
            with path.open("rb") as fh:
                data = pickle.load(fh)
        elif fmt == "json":
            with path.open("rt") as fh:
                data = json.load(fh)

        self.grain_IDs = np.array(data.pop("grain_IDs"))
        self.slice_data = unjsonify_dict(data.get("slice_data"))

        if "grain_IDs_periodic_image" in data:
            self.grain_IDs_periodic_image = np.array(data["grain_IDs_periodic_image"])

        if "seed_points" in data:
            self.seed_points = np.array(data["seed_points"])
            self.seed_points_args = unjsonify_dict(data["seed_points_args"])

        if "tessellated_sub_grain_IDs" in data:
            self.tessellated_sub_grain_IDs = np.array(data["tessellated_sub_grain_IDs"])
            self.is_dummy_seed = np.array(data["is_dummy_seed"])
            self.sub_grain_seeds_idx = [
                np.array(i) for i in data["sub_grain_seeds_idx"]
            ]
            self.is_interface_low_angle_GB = np.array(data["is_interface_low_angle_GB"])

        logger.info("Done.")

    def save(self, fmt="pickle"):
        path = {"json": self.JSON_path, "pickle": self.pickle_path}[fmt]
        logger.info(f"Saving {self.__class__.__name__!r} to file {str(path)}...")

        if not self.is_segmented:
            return ValueError("Not segmented yet; run `do_segmentation` first!")

        data = {
            "slice_data": jsonify_dict(self.slice_data),
            "grain_IDs": self.grain_IDs.tolist(),
        }
        if self.grain_IDs_periodic_image is not None:
            data["grain_IDs_periodic_image"] = self.grain_IDs_periodic_image.tolist()

        if self.seed_points is not None:
            data["seed_points"] = self.seed_points.tolist()
            data["seed_points_args"] = jsonify_dict(self.seed_points_args)

        if self.tessellated_sub_grain_IDs is not None:
            data["tessellated_sub_grain_IDs"] = self.tessellated_sub_grain_IDs.tolist()
            data["is_dummy_seed"] = self.is_dummy_seed.tolist()
            data["sub_grain_seeds_idx"] = [i.tolist() for i in self.sub_grain_seeds_idx]
            data["is_interface_low_angle_GB"] = self.is_interface_low_angle_GB.tolist()

        if fmt == "pickle":
            with path.open("wb") as fh:
                pickle.dump(data, fh)
        elif fmt == "json":
            with path.open("wt") as fh:
                json.dump(data, fh, indent=2)

        logger.info("Done.")

    def _estimate_sub_grain_size(self):
        rho = self.estimate_dislocation_density()
        K = 1_000
        cell_diameter = K / np.sqrt(rho)
        cell_density = 1 / cell_diameter**3

    def set_seed_points(self, method, pixel_length, redo=False, **kwargs):

        if self.seed_points is not None and not redo:
            logger.info("Seed points already set; exiting.")
            return

        self.seed_points_args = {
            "method": method,
            "pixel_length": pixel_length,
            **kwargs,
        }

        rng = np.random.default_rng(kwargs.get("random_seed"))

        if method == "random":

            number = kwargs.get("number")
            seeds = np.hstack(
                [
                    rng.integers(0, self.slice_grid_size[0], (number, 1)),
                    rng.integers(0, self.slice_grid_size[1], (number, 1)),
                ]
            )

        elif method == "dislocation_density":

            # TODO: link to relationship between cell size, L and rho (L ~= 1000/sqrt(rho)) - i.e. 10 microns
            # choose probs as 1/L**3 "cell density"
            # pixel size as ~0.5 micron? enough to resolve the sub grains

            # so we want an L array, then a cell density (i.e. probability array)
            # then make a random array (0-1) and choose cells if probability array is less than random array (?)

            rho = self.estimate_dislocation_density()

            K = 1_000
            cell_diameter = K / np.sqrt(rho)
            cell_diameter_normed = cell_diameter / pixel_length

            cell_density = 1 / cell_diameter_normed**2

            prob_field = cell_density.flatten()
            number = int(prob_field.sum())

            print(f"number of seed points set: {number}")

            prob_field /= prob_field.sum()  # probabilities array should sum to one

            sample_index = rng.choice(
                a=prob_field.size,
                p=prob_field,
                size=(number,),
                replace=False,  # to avoid coincident seed points!
            )

            # Take this index and adjust it so it matches the original array
            seeds = np.vstack(np.unravel_index(sample_index, cell_density.shape)).T[
                :, ::-1
            ]  # swap x-y

        self.seed_points = seeds

    def estimate_dislocation_density(
        self, alpha=0.3, shear_modulus=44e9, burgers_vector=0.3e-9
    ):
        """
        Notes
        -----
        Equation for dislocation density is Eq. 2.2. from Humphreys & Hatherly (2004),
        Chapter 2.

        """

        # Get the stress field slice:
        stress = self.element.outputs.volume_element_response["field_data"]["sigma_vM"][
            "data"
        ]
        stress = stress[self.slice_selection.increment_idx]
        stress_data = get_volumetric_slice(
            stress,
            x=self.slice_selection.x,
            y=self.slice_selection.y,
            z=self.slice_selection.z,
            eye=self.slice_selection.eye,
            up=self.slice_selection.up,
        )
        stress_slice = stress_data["slice_data"]

        # TODO: delta rho prop to delta stress?

        rho = (stress_slice / (alpha * shear_modulus * burgers_vector)) ** 2

        return rho

    def tessellate_seed_points(self):
        if self.seed_points is None:
            raise ValueError("Seed points not set yet; run `set_seed_points` first!")
        (
            new_sub_grain_seeds,
            sub_grain_seeds_idx,
            grain_coords,
            is_dummy_seed,
        ) = partition_sub_grain_seeds_into_grains(
            self.seed_points,
            self.grain_IDs,
        )

        sub_grain_IDs = tessellate_sub_grain_seeds(
            new_sub_grain_seeds,
            sub_grain_seeds_idx,
            self.slice_grid_size,
            grain_coords,
        )

        self.tessellated_sub_grain_IDs = sub_grain_IDs
        self.seed_points = new_sub_grain_seeds
        self.is_dummy_seed = is_dummy_seed
        self.sub_grain_seeds_idx = sub_grain_seeds_idx

        num_cipher_phases = len(np.unique(self.tessellated_sub_grain_IDs))
        print(f"num_cipher_phases: {num_cipher_phases}")

        is_low_angle_GB = np.zeros((num_cipher_phases, num_cipher_phases), dtype=int)
        for grain in self.sub_grain_seeds_idx:
            for row_idx, col_idx in combinations(grain, r=2):
                is_low_angle_GB[[row_idx, col_idx], [col_idx, row_idx]] = 1

        self.is_interface_low_angle_GB = is_low_angle_GB

    def show_estimated_dislocation_density(self):

        rho = self.estimate_dislocation_density()
        fig = graph_objects.FigureWidget(
            data=[
                {
                    "type": "heatmap",
                    "z": rho,
                    "xaxis": "x1",
                    "yaxis": "y1",
                    "coloraxis": "coloraxis",
                },
            ],
            layout={
                "height": 700,
                "coloraxis1": {
                    "colorscale": "viridis",
                },
                "xaxis": {
                    "scaleanchor": "y",
                    "constrain": "domain",
                },
                "yaxis": {
                    "autorange": "reversed",
                },
            },
        )
        return fig

    def show_tessellated_sub_grain_IDs(self):

        fig = graph_objects.FigureWidget(
            data=[
                {
                    "type": "heatmap",
                    "z": self.grain_IDs,
                    "xaxis": "x1",
                    "yaxis": "y1",
                    "coloraxis": "coloraxis",
                },
                {
                    "type": "heatmap",
                    "z": self.tessellated_sub_grain_IDs,
                    "xaxis": "x2",
                    "yaxis": "y2",
                    "coloraxis": "coloraxis2",
                },
                {
                    "type": "scatter",
                    "x": self.seed_points[~np.array(self.is_dummy_seed), 1],
                    "y": self.seed_points[~np.array(self.is_dummy_seed), 0],
                    "text": np.arange(self.seed_points.shape[0]),
                    "name": "sub grain seed points",
                    "mode": "markers",
                    "marker": {
                        "color": "red",
                        "size": 6,
                    },
                    "xaxis": "x2",
                    "yaxis": "y2",
                },
                {
                    "type": "scatter",
                    "x": self.seed_points[np.array(self.is_dummy_seed), 1],
                    "y": self.seed_points[np.array(self.is_dummy_seed), 0],
                    "text": np.arange(self.seed_points.shape[0]),
                    "name": "sub grain seed points (dummy)",
                    "mode": "markers",
                    "marker": {
                        "color": "red",
                        "size": 4,
                    },
                    "xaxis": "x2",
                    "yaxis": "y2",
                },
            ],
            layout={
                "height": 700,
                "coloraxis": {
                    "colorscale": "viridis",
                    "colorbar": {
                        "title": "Grain ID",
                        "x": 0,
                    },
                },
                "coloraxis2": {
                    "colorscale": "viridis",
                    "colorbar": {
                        "title": "Sub-grain ID",
                    },
                },
                "xaxis": {
                    "scaleanchor": "y",
                    "constrain": "domain",
                    "domain": [0.1, 0.45],
                    "title": "Clustered grain IDs",
                },
                "xaxis2": {
                    "scaleanchor": "y",
                    "constrain": "domain",
                    "domain": [0.55, 0.9],
                    "title": "Tessellated sub grain IDs",
                },
                "yaxis": {
                    "autorange": "reversed",
                },
                "yaxis2": {
                    "anchor": "x2",
                    "scaleanchor": "y",
                    "autorange": "reversed",
                },
            },
        )
        return fig

    def prepare_segmentation(self, redo=False):

        if self.is_segmented and not redo:
            logger.info("Already segmented; exiting.")
            return

        logger.info("Running segmentation...")
        self.slice_data = self._get_slice_data()

        self.seg_dir.mkdir(exist_ok=True)
        return self.seg_dir

    def write_CIPHER_input_data(self, path, materials, interfaces, solution_parameters):

        if self.seed_points is None:
            raise ValueError("Seed points not set yet; run `set_seed_points` first!")

        phase_map = self.get_field_data("phase")

        # Scale to a power of 2:
        scaled_grid_size = 2**8
        zoom_factor = scaled_grid_size / self.slice_grid_size[0]
        scaled_sub_grain_IDs = zoom(
            self.tessellated_sub_grain_IDs, zoom_factor, order=0
        )
        scaled_phase_map = zoom(phase_map, zoom_factor, order=0)
        scaled_pixel_length = self.seed_points_args["pixel_length"] / zoom_factor

        cipher_grid_size = [int(i * zoom_factor) for i in self.slice_grid_size]
        cipher_size = [i * scaled_pixel_length for i in cipher_grid_size]

        cipher_voxel_phase_mapping = np.reshape(scaled_sub_grain_IDs, -1, order="F")
        phase_map_flat = np.reshape(scaled_phase_map, -1, order="F")

        unq, ind = np.unique(cipher_voxel_phase_mapping, return_index=True)
        num_cipher_phases = len(unq)
        cipher_phase_material_mapping = phase_map_flat[ind]

        cipher_materials = np.array(
            self.element.outputs.volume_element_response["field_data"]["phase"]["meta"][
                "phase_names"
            ]
        )

        (cipher_interfaces_mat_idx, cipher_interface_mapping) = get_interface_map(
            cipher_phase_material_mapping,
            split_interface=self.is_interface_low_angle_GB,
        )

        split_interfaces = ["HAGB", "LAGB"]

        cipher_interfaces = []
        seen_interface_idx = []
        for interface in cipher_materials[cipher_interfaces_mat_idx]:
            split_interface_str = ""
            if interface[0] == interface[1]:
                if interface.tolist() not in seen_interface_idx:
                    split_interface_str = "-" + split_interfaces[0]
                    seen_interface_idx.append(interface.tolist())
                else:
                    split_interface_str = "-" + split_interfaces[1]

            cipher_interfaces.append("-".join(interface) + f"{split_interface_str}")

        maxnrefine = 10
        max_grid_size = 2**maxnrefine

        interface_scale = 4  # 4 to 8
        interfacewidth = interface_scale * max(cipher_size) / max_grid_size

        if set(materials.keys()) != set(cipher_materials):
            raise ValueError(
                f"`materials` must be a dict with keys: {cipher_materials!r}"
            )

        if set(interfaces.keys()) != set(cipher_interfaces):
            raise ValueError(
                f"`interfaces` must be a dict with keys: {cipher_interfaces!r}"
            )

        sln_params = copy.deepcopy(solution_parameters)
        if sln_params.get("interfacewidth") is None:
            sln_params["interfacewidth"] = interfacewidth
        if sln_params.get("maxnrefine") is None:
            sln_params["maxnrefine"] = maxnrefine

        cipher_input_data = {
            "header": {
                "grid": cipher_grid_size,
                "size": cipher_size,
                "n_phases": num_cipher_phases,
                "materials": [str(k) for k in cipher_materials],
                "interfaces": [str(k) for k in cipher_interfaces],
                "components": ["ti"],
                "outputs": [
                    "phaseid",
                    "matid",  # alpha and beta phase
                    "interfaceid",  # zeros in bulk, colour-coded at interfaces
                ],
            },
            "solution_parameters": dict(sorted(sln_params.items())),
            "material": materials,
            "interface": interfaces,
            "mappings": {
                "phase_material_mapping": LiteralScalarString(
                    get_compressed_1D_array_string(cipher_phase_material_mapping + 1)
                    + "\n"
                ),
                "voxel_phase_mapping": LiteralScalarString(
                    get_compressed_1D_array_string(cipher_voxel_phase_mapping + 1)
                    + "\n"
                ),
                "interface_mapping": LiteralScalarString(
                    get_compressed_1D_array_string(
                        cipher_interface_mapping.flatten(order="C") + 1
                    )
                    + "\n"
                ),
            },
        }

        yaml = YAML()
        with Path(path).open("w") as fp:
            yaml.dump(cipher_input_data, fp)


class MTEX_FMC_Clusterer(Clusterer):
    def format_segmentations_directory_name(self):
        parametrised_path = super().format_segmentations_directory_name()
        parametrised_path += (
            f"__C_Maha_{self.parameters['C_Maha']:.2f}"
            f"__smoothing_{self.parameters['smoothing']}"
        )
        return parametrised_path

    def _get_seg_dir(self):
        return self.pre_processor.segmentation_sub_dir.joinpath(
            self.format_segmentations_directory_name()
        )

    def do_segmentation(self, redo=False):

        if self.prepare_segmentation(redo) is None:
            return

        params = copy.deepcopy(self.parameters)

        if "MTEX_script_path" not in params:
            params["MTEX_script_path"] = "segment_slice.m"

        if "specimen_symmetry" not in params:
            params["specimen_symmetry"] = "triclinic"

        # DAMASK uses P=-1 convention:
        eulers_flat = quat2euler(self.slice_data["quats_flat"], degrees=True, P=-1)

        # Write orientations to "EBSD" file for MTEX to load:
        ori_data_path = self.seg_dir.joinpath("MTEX_EBSD_2D_slice.txt")
        write_MTEX_EBSD_file(
            coords=self.slice_data["coords_flat"],
            euler_angles=eulers_flat,
            phases=self.slice_data["phases_flat"],
            filename=ori_data_path,
        )

        script_path = Path(params["MTEX_script_path"])
        copied_script_path = self.seg_dir.joinpath(script_path.name)
        copied_script_path.write_bytes(script_path.read_bytes())

        fig_resolution = 50
        if self.slice_selection.is_periodic:
            fig_resolution *= 3

        phase_syms = {k: v["lattice"] for k, v in self.element.inputs.phases.items()}
        sym_lookup = {
            "hP": "hexagonal",
            "cI": "cubic",
        }
        crys_syms = [
            {
                "symmetry": sym_lookup[phase_syms[phase_name]],
                "mineral": phase_name,
                "unit_cell_alignment": {"x": "a"},  # DAMASK alignment
            }
            for phase_name in self.slice_data["phase_names"]
        ]

        params["crystal_symmetries"] = crys_syms

        args_JSON_path = self.seg_dir.joinpath("MTEX_args.json")
        working_dir = str(self.seg_dir)
        MTEX_data = {
            **params,
            "fig_resolution": fig_resolution,
            "orientations_data_path": str(ori_data_path),
            "working_dir": working_dir,
        }
        write_MTEX_JSON_file(data=MTEX_data, filename=args_JSON_path)

        logger.info("Running MTEX script...")
        matlab_grain_IDs = run_matlab_script(
            script_name=str(script_path.stem),
            script_dir=working_dir,
            args=[str(args_JSON_path)],
            nargout=1,
        )
        grain_IDs = np.array(matlab_grain_IDs)

        logger.info("Finished MTEX script.")
        self.grain_IDs = grain_IDs

        self.do_segmentation_post_processing()
        self.save()
        logger.info("Finished.")

    def do_segmentation_post_processing(self):

        logger.info("Starting post-processing...")

        if self.slice_selection.is_periodic:
            logger.info("Accounting for periodicity...")

            self.grain_IDs_periodic_image = self.grain_IDs
            grain_IDs = self.grain_IDs

            view_offset = 0
            im_cropped_size = [
                int(grain_IDs.shape[0] / 3),
                int(grain_IDs.shape[1] / 3),
            ]
            grain_IDs = grain_IDs[
                im_cropped_size[0]
                - view_offset : int(grain_IDs.shape[0] * 2 / 3)
                - view_offset,
                im_cropped_size[1]
                - view_offset : int(grain_IDs.shape[1] * 2 / 3)
                - view_offset,
            ]

            grain_IDs = remap_periodic_boundary_IDs(
                grain_IDs,
                grain_IDs[0],
                grain_IDs[-1],
            )
            grain_IDs = remap_periodic_boundary_IDs(
                grain_IDs,
                grain_IDs[:, 0],
                grain_IDs[:, -1],
            )
            _, inv = np.unique(grain_IDs.reshape(-1), return_inverse=True)
            grain_IDs = inv.reshape(grain_IDs.shape)
            self.grain_IDs = grain_IDs
