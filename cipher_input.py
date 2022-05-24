from multiprocessing.sharedctypes import Value
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from random import random
from typing import Optional, List, Union, Tuple, Dict

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from discrete_voronoi import DiscreteVoronoi
from voxel_map import VoxelMap


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
    def __init__(
        self,
        phase_material,
        material_names,
        interfaces,
        size,
        seeds=None,
        voxel_phase=None,
        voxel_map=None,
    ):

        if sum(i is not None for i in (voxel_phase, voxel_map)) != 1:
            raise ValueError(f"Specify exactly one of `voxel_phase` and `voxel_map`")

        if voxel_map is None:
            voxel_map = VoxelMap(region_ID=voxel_phase, size=size, is_periodic=True)
        else:
            voxel_phase = voxel_map.region_ID

        self.voxel_phase = voxel_phase
        self.voxel_map = voxel_map

        self.phase_material = phase_material
        self.material_names = material_names
        self.interfaces = interfaces
        self.size = np.asarray(size)
        self.seeds = seeds

        if self.size.size != self.dimension:
            raise ValueError(
                f"`size` ({self.size}) implies {self.size.size} dimensions, but "
                f"`voxel_phase` implies {self.voxel_phase.dimension} dimensions."
            )

        for idx, i in enumerate(self.interfaces):
            i.index = idx

        self._interface_map = self._get_interface_map()

    @property
    def dimension(self):
        return self.voxel_map.dimension

    @property
    def grid_size(self):
        return np.array(self.voxel_map.grid_size)

    @property
    def neighbour_voxels(self):
        return self.voxel_map.neighbour_voxels

    @property
    def neighbour_list(self):
        return self.voxel_map.neighbour_list

    @property
    def voxel_material(self):
        return self.phase_material[self.voxel_phase]

    @property
    def interface_names(self):
        return [i.name for i in self.interfaces]

    @property
    def num_phases(self):
        return len(self.phase_material)

    @property
    def interface_map(self):
        return self._interface_map

    @property
    def seeds_grid(self):
        return np.round(self.grid_size * self.seeds / self.size, decimals=0).astype(int)

    @staticmethod
    def get_unique_random_seeds(num_phases, size, grid_size, random_seed=None):
        return DiscreteVoronoi.get_unique_random_seeds(
            num_regions=num_phases,
            size=size,
            grid_size=grid_size,
            random_seed=random_seed,
        )

    @staticmethod
    def assign_phase_material_randomly(
        num_materials,
        num_phases,
        volume_fractions,
        random_seed=None,
    ):

        print(
            "Randomly assigning phases to materials according to volume_fractions...",
            end="",
        )
        rng = np.random.default_rng(seed=random_seed)
        phase_material = rng.choice(
            a=num_materials,
            size=num_phases,
            p=volume_fractions,
        )
        print("done!")
        return phase_material

    @classmethod
    def from_voronoi(
        cls,
        volume_fractions,
        interfaces,
        material_names,
        grid_size,
        size,
        seeds=None,
        num_phases=None,
        random_seed=None,
    ):

        if np.sum(volume_fractions) != 1:
            raise ValueError("`volume_fractions` must sum to 1.")

        if sum(i is not None for i in (seeds, num_phases)) != 1:
            raise ValueError(f"Specify exactly one of `seeds` and `num_phases`")

        if seeds is None:
            vor_map = DiscreteVoronoi.from_random(
                num_regions=num_phases,
                grid_size=grid_size,
                size=size,
                is_periodic=True,
                random_seed=random_seed,
            )

        else:
            vor_map = DiscreteVoronoi.from_seeds(
                region_seeds=seeds,
                grid_size=grid_size,
                size=size,
                is_periodic=True,
            )

        num_materials = len(volume_fractions)

        phase_material = cls.assign_phase_material_randomly(
            num_materials=num_materials,
            num_phases=vor_map.num_regions,
            volume_fractions=volume_fractions,
            random_seed=random_seed,
        )

        return cls(
            voxel_map=vor_map,
            phase_material=phase_material,
            material_names=material_names,
            interfaces=interfaces,
            size=size,
            seeds=seeds,
        )

    @classmethod
    def from_seed_voronoi(
        cls,
        seeds,
        volume_fractions,
        interfaces,
        material_names,
        grid_size,
        size,
        random_seed=None,
    ):
        return cls.from_voronoi(
            volume_fractions=volume_fractions,
            interfaces=interfaces,
            material_names=material_names,
            grid_size=grid_size,
            size=size,
            seeds=seeds,
            random_seed=random_seed,
        )

    @classmethod
    def from_random_voronoi(
        cls,
        num_phases,
        volume_fractions,
        interfaces,
        material_names,
        grid_size,
        size,
        random_seed=None,
    ):
        return cls.from_voronoi(
            volume_fractions=volume_fractions,
            interfaces=interfaces,
            material_names=material_names,
            grid_size=grid_size,
            size=size,
            num_phases=num_phases,
            random_seed=random_seed,
        )

    @property
    def volume_fractions(self):
        _, frequency = np.unique(self.phase_material, return_counts=True)
        return frequency / self.num_phases

    def get_interface_idx(self):
        return self.voxel_map.get_interface_idx(self.interface_map)

    def get_interface_map_indices(self, mat_A, mat_B):
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

        print("Finding interface map matrix...", end="")

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
            manual_set = [i.phase_pairs is not None for i in int_defs]
            any_manual_set = any(manual_set)
            all_manual_set = all(manual_set)
            if any_frac_set:
                if any_manual_set:
                    raise ValueError(
                        f"For interface {mat_pair}, specify phase pairs manually for all "
                        f"defined interfaces using `phase_pairs`, or specify `type_fraction`"
                        f"for all defined interfaces. You cannot mix them."
                    )

            all_phase_pairs = self.get_interface_map_indices(*mat_pair)
            if any_manual_set:
                if not all_manual_set:
                    raise ValueError(
                        f"For interface {mat_pair}, specify phase pairs manually for all "
                        f"defined interfaces using `phase_pairs`, or specify `type_fraction`"
                        f"for all defined interfaces. You cannot mix them."
                    )

                phase_pairs_by_type = {i.type_label: i.phase_pairs for i in int_defs}

                # check that given phase_pairs combine to the set of all phase_pairs
                # for this material-material pair:
                all_given_phase_pairs = np.hstack([i.phase_pairs for i in int_defs])

                # sort by first-phase, then second-phase, for comparison:
                srt = np.lexsort(all_given_phase_pairs[::-1])
                all_given_phase_pairs = all_given_phase_pairs[:, srt]

                if all_given_phase_pairs.shape != all_phase_pairs.shape or not np.all(
                    all_given_phase_pairs == all_phase_pairs
                ):
                    raise ValueError(
                        f"Missing `phase_pairs` for interface {mat_pair}. The following "
                        f"phase pairs must all be included for this interface: "
                        f"{all_phase_pairs}"
                    )

                for int_i in int_defs:
                    phase_pairs_i = int_i.phase_pairs
                    int_map[phase_pairs_i[0], phase_pairs_i[1]] = int_i.index

                    if not upper_tri_only:
                        int_map[phase_pairs_i[1], phase_pairs_i[0]] = int_i.index

            else:
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

        print("done!")

        return int_map


@dataclass
class CIPHERInput:
    geometry: CIPHERGeometry
    materials: Dict
    components: List
    outputs: List
    solution_parameters: Dict

    @classmethod
    def from_voronoi(
        cls,
        grid_size,
        size,
        volume_fractions,
        materials,
        interfaces,
        components,
        outputs,
        solution_parameters,
        seeds=None,
        num_phases=None,
        random_seed=None,
    ):
        if len(volume_fractions) != len(materials):
            raise ValueError(
                f"`volume_fractions` (length {len(volume_fractions)}) must be of equal "
                f"length to `materials` (length {len(materials)})."
            )

        geometry = CIPHERGeometry.from_voronoi(
            num_phases=num_phases,
            seeds=seeds,
            volume_fractions=volume_fractions,
            interfaces=interfaces,
            material_names=list(materials.keys()),
            grid_size=grid_size,
            size=size,
            random_seed=random_seed,
        )

        inp = cls(
            geometry=geometry,
            materials=materials,
            components=components,
            outputs=outputs,
            solution_parameters=solution_parameters,
        )
        return inp

    @classmethod
    def from_seed_voronoi(
        cls,
        seeds,
        grid_size,
        size,
        volume_fractions,
        materials,
        interfaces,
        components,
        outputs,
        solution_parameters,
        random_seed=None,
    ):

        return cls.from_voronoi(
            seeds=seeds,
            grid_size=grid_size,
            size=size,
            volume_fractions=volume_fractions,
            materials=materials,
            interfaces=interfaces,
            components=components,
            outputs=outputs,
            solution_parameters=solution_parameters,
            random_seed=random_seed,
        )

    @classmethod
    def from_random_voronoi(
        cls,
        num_phases,
        grid_size,
        size,
        volume_fractions,
        materials,
        interfaces,
        components,
        outputs,
        solution_parameters,
        random_seed=None,
    ):

        return cls.from_voronoi(
            num_phases=num_phases,
            grid_size=grid_size,
            size=size,
            volume_fractions=volume_fractions,
            materials=materials,
            interfaces=interfaces,
            components=components,
            outputs=outputs,
            solution_parameters=solution_parameters,
            random_seed=random_seed,
        )

    @classmethod
    def from_voxel_phase_map(
        cls,
        voxel_phase,
        size,
        materials,
        interfaces,
        components,
        outputs,
        solution_parameters,
        random_seed=None,
        volume_fractions=None,
        phase_material=None,
    ):
        if sum(i is not None for i in (volume_fractions, phase_material)) != 1:
            raise ValueError(
                f"Specify exactly one of `phase_material` and `volume_fractions`"
            )

        if volume_fractions is not None:
            if np.sum(volume_fractions) != 1:
                raise ValueError("`volume_fractions` must sum to 1.")

            if len(volume_fractions) != len(materials):
                raise ValueError(
                    f"`volume_fractions` (length {len(volume_fractions)}) must be of equal "
                    f"length to `materials` (length {len(materials)})."
                )
            num_phases = np.unique(voxel_phase).size
            num_materials = len(volume_fractions)

            phase_material = CIPHERGeometry.assign_phase_material_randomly(
                num_materials=num_materials,
                num_phases=num_phases,
                volume_fractions=volume_fractions,
                random_seed=random_seed,
            )

        geometry = CIPHERGeometry(
            voxel_phase=voxel_phase,
            phase_material=phase_material,
            material_names=list(materials.keys()),
            interfaces=interfaces,
            size=size,
        )
        inp = cls(
            geometry=geometry,
            materials=materials,
            components=components,
            outputs=outputs,
            solution_parameters=solution_parameters,
        )
        return inp

    def get_header(self):
        out = {
            "grid": self.geometry.grid_size.tolist(),
            "size": self.geometry.size.tolist(),
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
