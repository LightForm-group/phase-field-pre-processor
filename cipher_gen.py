from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Dict

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from discrete_voronoi import DiscreteVoronoi


def generate_voxel_phase_map(num_phases, grid_size, size):
    rng = np.random.default_rng()
    seeds = np.hstack(
        [
            rng.integers(0, grid_size[0], (num_phases, 1)),
            rng.integers(0, grid_size[1], (num_phases, 1)),
        ]
    )
    if np.array(grid_size).ndim == 3:
        seeds = np.hstack([seeds, rng.integers(0, grid_size[2], (num_phases, 1))])

    vor = DiscreteVoronoi(seeds=seeds, grid_size=grid_size, size=size, periodic=True)
    voxel_phase = vor.voxel_assignment
    return voxel_phase


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


def compress_1D_array_string(arr, item_delim="\n"):
    out = []
    for n, v in zip(*compress_1D_array(arr)):
        out.append(f"{n} of {v}" if n > 1 else f"{v}")

    return item_delim.join(out)


def decompress_1D_array_string(arr_str, item_delim="\n"):
    out = []
    for i in arr_str.split(item_delim):
        if "of" in i:
            n, i = i.split("of")
            i = [int(i.strip()) for _ in range(int(n.strip()))]
        else:
            i = [int(i.strip())]
        out.extend(i)
    return np.array(out)


@dataclass
class InterfaceDefinition:
    """
    Attributes
    ----------
    materials :
        Between which named materials this interface applies.
    type_label :
        To distinguish between multiple interfaces that all apply between the same pair of
        materials
    phase_pairs :
        List of phase pair indices that should have this interface type (for manual
        specification)
    """

    materials: Union[List[str], Tuple[str]]
    properties: Dict
    name: Optional[str] = None
    type_label: Optional[str] = ""
    type_fraction: Optional[float] = None
    phase_pairs: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.materials[0]}-{self.materials[1]}{f'-{self.type_label}' if self.type_label else ''}"

        if self.type_fraction is not None and self.phase_pairs is not None:
            raise ValueError("Specify either `type_fraction` or `phase_pairs`.")


class CIPHERGeometry:
    def __init__(self, voxel_phase, phase_material, material_names, interfaces, size):
        self.voxel_phase = voxel_phase
        self.phase_material = phase_material
        self.material_names = material_names
        self.interfaces = interfaces
        self.size = size

        for idx, i in enumerate(self.interfaces):
            i.index = idx

        self._interface_map = self._get_interface_map()

    @property
    def grid_size(self):
        return self.voxel_phase.shape

    @property
    def interface_names(self):
        return [i.name for i in self.interfaces]

    @property
    def num_phases(self):
        return len(self.phase_material)

    @property
    def interface_map(self):
        return self._interface_map

    @classmethod
    def from_random_voronoi(
        cls, num_phases, volume_fractions, interfaces, material_names, grid_size, size
    ):

        voxel_phase = generate_voxel_phase_map(num_phases, grid_size, size)
        if not np.sum(volume_fractions) == 1:
            raise ValueError("`volume_fractions` must sum to 1.")

        num_materials = len(volume_fractions)
        phase_material = np.random.choice(
            a=num_materials, size=num_phases, p=volume_fractions
        )

        return cls(
            voxel_phase=voxel_phase,
            phase_material=phase_material,
            material_names=material_names,
            interfaces=interfaces,
            size=size,
        )

    @property
    def volume_fractions(self):
        _, frequency = np.unique(self.phase_material, return_counts=True)
        return frequency / self.num_phases

    def _get_interface_map_indices(self, mat_A, mat_B):
        """Get an array of integer indices that index the (upper triangle of the) 2D
        symmetric interface map array, corresponding to a given material pair."""

        # First get phase indices belonging to the two materials:
        mat_A_idx = self.material_names.index(mat_A)
        mat_B_idx = self.material_names.index(mat_B)

        mat_A_phase_idx = np.where(self.phase_material == mat_A_idx)[0]
        mat_B_phase_idx = np.where(self.phase_material == mat_B_idx)[0]

        A_idx = np.repeat(mat_A_phase_idx, mat_B_phase_idx.shape[0])
        B_idx = np.tile(mat_B_phase_idx, mat_A_phase_idx.shape[0])

        map_idx = np.vstack((A_idx, B_idx))
        map_idx_srt = np.sort(map_idx, axis=0)  # map onto upper triangle
        map_idx_uniq = np.unique(map_idx_srt, axis=1)  # get unique pairs only

        # remove diagonal elements (a phase can't have an interface with itself)
        map_idx_non_trivial = map_idx_uniq[:, map_idx_uniq[0] != map_idx_uniq[1]]

        return map_idx_non_trivial

    def _get_interface_map(self, upper_tri_only=False):

        int_map = np.zeros((self.num_phases, self.num_phases), dtype=int)

        ints_by_mat_pair = {}
        for int_def in self.interfaces:
            if int_def.materials not in ints_by_mat_pair:
                ints_by_mat_pair[int_def.materials] = []
            ints_by_mat_pair[int_def.materials].append(int_def)

        for mat_pair, int_defs in ints_by_mat_pair.items():

            names = [i.name for i in int_defs]
            if len(set(names)) < len(names):
                raise ValueError(
                    f"Multiple interface definitions for material pair "
                    f"{mat_pair} have the same `type_label`."
                )
            type_fracs = [i.type_fraction for i in int_defs]
            any_frac_set = any(i is not None for i in type_fracs)
            any_manual_set = any(i.phase_pairs is not None for i in int_defs)
            if any_frac_set:
                if any_manual_set:
                    raise ValueError(
                        f"For interface {mat_pair}, specify phase pairs manually for all "
                        f"defined interfaces using `phase_pairs`, or specify `type_fraction`"
                        f"for all defined interfaces. You cannot mix them."
                    )

            if not any_manual_set:
                # set default type fractions if missing
                remainder_frac = 1 - sum(i for i in type_fracs if i is not None)
                if remainder_frac > 0:
                    num_missing_type_frac = sum(1 for i in type_fracs if i is None)
                    if num_missing_type_frac == 0:
                        raise ValueError(
                            f"For interface {mat_pair}, `type_fraction` for all "
                            f"defined interfaces must sum to one."
                        )
                    remainder_frac_each = remainder_frac / num_missing_type_frac
                    for i in int_defs:
                        if i.type_fraction is None:
                            i.type_fraction = remainder_frac_each

                type_fracs = [i.type_fraction for i in int_defs]
                if sum(type_fracs) != 1:
                    raise ValueError(
                        f"For interface {mat_pair}, `type_fraction` for all "
                        f"defined interfaces must sum to one."
                    )

                # assign phase_pairs according to type fractions:
                all_phase_pairs = self._get_interface_map_indices(*mat_pair)
                num_pairs = all_phase_pairs.shape[1]
                type_nums_each = [round(i * num_pairs) for i in type_fracs]
                type_nums = np.cumsum(type_nums_each)
                if num_pairs % 2 == 1:
                    type_nums += 1

                shuffle_idx = np.random.choice(num_pairs, size=num_pairs, replace=False)
                phase_pairs_shuffled = all_phase_pairs[:, shuffle_idx]
                phase_pairs_split = np.split(phase_pairs_shuffled, type_nums, axis=1)[
                    :-1
                ]

                for idx, int_i in enumerate(int_defs):
                    phase_pairs_i = phase_pairs_split[idx]
                    int_map[phase_pairs_i[0], phase_pairs_i[1]] = int_i.index

                    if not upper_tri_only:
                        int_map[phase_pairs_i[1], phase_pairs_i[0]] = int_i.index

        return int_map


@dataclass
class CIPHERInput:
    geometry: CIPHERGeometry
    materials: Dict
    components: List
    outputs: List
    solution_parameters: Dict

    def get_header(self):
        out = {
            "grid": self.geometry.grid_size,
            "size": self.geometry.size,
            "n_phases": self.geometry.num_phases,
            "materials": self.geometry.material_names,
            "interfaces": self.geometry.interface_names,
            "components": self.components,
            "outputs": self.outputs,
        }
        return out

    def get_interfaces(self):
        return {i.name: i.properties for i in self.geometry.interfaces}

    def write_yaml(self, path):
        """Write the CIPHER input YAML file."""
        cipher_input_data = {
            "header": self.get_header(),
            "solution_parameters": dict(sorted(self.solution_parameters.items())),
            "material": self.materials,
            "interface": self.get_interfaces(),
            "mappings": {
                "phase_material_mapping": LiteralScalarString(
                    compress_1D_array_string(self.geometry.phase_material + 1) + "\n"
                ),
                "voxel_phase_mapping": LiteralScalarString(
                    compress_1D_array_string(
                        self.geometry.voxel_phase.flatten(order="F") + 1
                    )
                    + "\n"
                ),
                "interface_mapping": LiteralScalarString(
                    compress_1D_array_string(self.geometry.interface_map.flatten() + 1)
                    + "\n"
                ),
            },
        }

        yaml = YAML()
        path = Path(path)
        with path.open("wt") as fp:
            yaml.dump(cipher_input_data, fp)

        return path


def generate_CIPHER_input(
    materials,
    volume_fractions,
    num_phases,
    grid_size,
    size,
    interfaces,
    components,
    outputs,
    solution_parameters,
):
    """
    Parameters
    ----------

    interfaces :
        multiple types of the same phase-phase interface could be assigned in different ways:
        randomly, some number fraction, manually (i.e. by phase index),

    """

    if len(volume_fractions) != len(materials):
        raise ValueError(
            f"`volume_fractions` (length {len(volume_fractions)}) must be of equal length "
            f"to `materials` (length {len(materials)})."
        )

    geometry = CIPHERGeometry.from_random_voronoi(
        num_phases=num_phases,
        volume_fractions=volume_fractions,
        interfaces=interfaces,
        material_names=list(materials.keys()),
        grid_size=grid_size,
        size=size,
    )

    cipher_inputs = CIPHERInput(
        geometry=geometry,
        materials=materials,
        components=components,
        outputs=outputs,
        solution_parameters=solution_parameters,
    )

    return cipher_inputs
