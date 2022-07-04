import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Dict

import numpy as np
import pyvista as pv
import h5py
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from discrete_voronoi import DiscreteVoronoi
from utilities import set_by_path
from voxel_map import VoxelMap
from quats import quat_angle_between


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


class InterfaceDefinition:
    """
    Attributes
    ----------
    materials :
        Between which named materials this interface applies.  Specify this or `phase_types`.
    phase_types :
        Between which named phase types this interface applies. Specify this or `materials`.
    type_label :
        To distinguish between multiple interfaces that all apply between the same pair of
        materials
    phase_pairs :
        List of phase pair indices that should have this interface type (for manual
        specification). Can be specified as an (N, 2) array.
    """

    def __init__(
        self,
        properties: Dict,
        materials: Optional[Union[List[str], Tuple[str]]] = None,
        phase_types: Optional[Union[List[str], Tuple[str]]] = None,
        type_label: Optional[str] = "",
        type_fraction: Optional[float] = None,
        phase_pairs: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ):
        self._is_phase_pairs_set = False
        self.index = None  # assigned by parent CIPHERGeometry

        self.properties = properties
        self.materials = materials
        self.phase_types = phase_types
        self.type_label = type_label
        self.type_fraction = type_fraction
        self.phase_pairs = phase_pairs
        self.metadata = metadata

        self._validate()

    @property
    def is_phase_pairs_set(self):
        return self._is_phase_pairs_set

    @property
    def name(self):
        return self.get_name(self.phase_types, self.type_label)

    @property
    def phase_pairs(self):
        return self._phase_pairs

    @phase_pairs.setter
    def phase_pairs(self, phase_pairs):

        if phase_pairs is not None:
            self._is_phase_pairs_set = True

        if phase_pairs is None or len(phase_pairs) == 0:
            phase_pairs = np.array([]).reshape((0, 2))
        else:
            phase_pairs = np.asarray(phase_pairs)

        if phase_pairs.shape[1] != 2:
            raise ValueError(
                f"phase_pairs should be specified as an (N, 2) array or a list of "
                f"two-element lists, but has shape: {phase_pairs.shape}."
            )

        # sort so first index is smaller:
        phase_pairs = np.sort(phase_pairs, axis=1)

        # sort by first phase index, then by second phase-idx:
        srt = np.lexsort(phase_pairs.T[::-1])
        phase_pairs = phase_pairs[srt]

        self._phase_pairs = phase_pairs

    @property
    def num_phase_pairs(self):
        return self.phase_pairs.shape[0]

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if metadata is not None:
            for k, v in metadata.items():
                if len(v) != self.num_phase_pairs:
                    raise ValueError(
                        f"Item {k!r} in the `metadata` dict must have length equal to the "
                        f"number of phase pairs ({self.num_phase_pairs}) but has length: "
                        f"{len(v)}."
                    )
        self._metadata = metadata

    @staticmethod
    def get_name(phase_types, type_label):
        return (
            f"{phase_types[0]}-{phase_types[1]}"
            f"{f'-{type_label}' if type_label else ''}"
        )

    def _validate(self):
        if self.materials:
            if self.phase_types:
                raise ValueError(
                    "Specify exactly one of `materials` and `phase_types`."
                )
            self.phase_types = copy.copy(self.materials)

        elif not self.phase_types:
            raise ValueError("Specify exactly one of `materials` and `phase_types`.")

        if self.type_fraction is not None and self.phase_pairs.size:
            raise ValueError("Specify either `type_fraction` or `phase_pairs`.")


class MaterialDefinition:
    """Class to represent a material within a CIPHER simulation."""

    def __init__(
        self,
        name,
        properties,
        phase_types=None,
        target_volume_fraction=None,
        phases=None,
    ):

        self.name = name
        self.properties = properties
        self.target_volume_fraction = target_volume_fraction
        self._geometry = None

        if target_volume_fraction is not None and phases is not None:
            raise ValueError(
                f"Cannot specify both `target_volume_fraction` and `phases` for material "
                f"{self.name!r}."
            )  # TODO: test raise

        if target_volume_fraction is not None:
            if target_volume_fraction == 0.0 or target_volume_fraction > 1.0:
                raise ValueError(
                    f"Target volume fraction must be greater than zero and less than or "
                    f"equal to one, but specified value for material {self.name!r} was "
                    f"{target_volume_fraction!r}."
                )  # TODO: test raise

        if phases is not None:
            for i in phase_types or []:
                if i.phases is not None:
                    raise ValueError(
                        f"Cannot specify `phases` in any of the phase type definitions if "
                        f"`phases` is also specified in the material definition."
                    )  # TODO: test raise
        else:
            if phase_types:
                is_phases_given = [i.phases is not None for i in phase_types]
                if any(is_phases_given) and sum(is_phases_given) != len(phase_types):
                    raise ValueError(
                        f"If specifying `phases` for a phase type for material "
                        f"{self.name!r}, `phases` must be specified for all phase types."
                    )  # TODO: test raise

        if phase_types is None:
            phase_types = [PhaseTypeDefinition(phases=phases)]

        if len(phase_types) > 1:
            pt_labels = [i.type_label for i in phase_types]
            if len(set(pt_labels)) < len(pt_labels):
                raise ValueError(
                    f"Phase types belonging to the same material ({self.name!r}) must have "
                    f"distinct `type_label`s."
                )  # TODO: test raise

        self.phase_types = phase_types

        if self.target_volume_fraction is not None:
            if self.phases is not None:
                raise ValueError(
                    f"Cannot specify both `target_volume_fraction` and `phases` for "
                    f"material {self.name!r}."
                )  # TODO: test raise

        is_type_frac = [i.target_type_fraction is not None for i in phase_types]
        if phase_types[0].phases is None:
            num_unassigned_vol = self.num_phase_types - sum(is_type_frac)
            assigned_vol = sum(i or 0.0 for i in self.target_phase_type_fractions)
            if num_unassigned_vol:
                frac = (1.0 - assigned_vol) / num_unassigned_vol
                if frac <= 0.0:
                    raise ValueError(
                        f"All phase type target volume fractions must sum to one, but "
                        f"assigned target volume fractions sum to {assigned_vol} with "
                        f"{num_unassigned_vol} outstanding unassigned phase type volume "
                        f"fraction(s)."
                    )  # TODO: test raise
            for i in self.phase_types:
                if i.target_type_fraction is None:
                    i.target_type_fraction = frac

            assigned_vol = sum(self.target_phase_type_fractions)
            if not np.isclose(assigned_vol, 1.0):
                raise ValueError(
                    f"All phase type target type fractions must sum to one, but target "
                    f"type fractions sum to {assigned_vol}."
                )  # TODO: test raise

        for i in self.phase_types:
            i._material = self

    @property
    def geometry(self):
        return self._geometry

    @property
    def num_phase_types(self):
        return len(self.phase_types)

    @property
    def target_phase_type_fractions(self):
        return [i.target_type_fraction for i in self.phase_types]

    @property
    def phases(self):
        try:
            return np.concatenate([i.phases for i in self.phase_types])
        except ValueError:
            # phases not yet assigned
            return None

    @property
    def index(self):
        """Get the index within the geometry materials list."""
        return self.geometry.materials.index(self)

    @property
    def phase_type_fractions(self):
        """Get the actual type volume (voxel) fractions within the material."""
        phase_type_fractions = []
        for i in self.phase_types:
            num_mat_voxels = self.geometry.material_num_voxels[self.index]
            pt_num_voxels = np.sum(self.geometry.phase_num_voxels[i.phases])
            phase_type_fractions.append(pt_num_voxels / num_mat_voxels)
        return np.array(phase_type_fractions)

    def assign_phases(self, phases, random_seed=None):
        """Assign given phase indices to phase types according to target_type_fractions."""

        phases = np.asarray(phases)

        # Now assign phases:
        rng = np.random.default_rng(seed=random_seed)
        phase_phase_type = rng.choice(
            a=self.num_phase_types,
            size=phases.size,
            p=self.target_phase_type_fractions,
        )
        for type_idx, phase_type in enumerate(self.phase_types):
            phase_type.phases = phases[np.where(phase_phase_type == type_idx)[0]]


class PhaseTypeDefinition:
    """Class to represent a type of phase (i.e. grain) within a CIPHER material.

    Attributes
    ----------
    material : MaterialDefinition
        Material to which this phase type belongs.
    type_label : str
        To distinguish between multiple phase types that all belong to the same material.
    target_type_fraction : float
    phases : ndarray of shape (N,) of int
        Phases that belong to this phase type.
    orientations : ndarray of shape (N, 4) of float
        Quaternion orientations for each phase.
    """

    def __init__(
        self,
        type_label=None,
        target_type_fraction=None,
        phases=None,
        orientations=None,
    ):
        self.type_label = type_label
        self.target_type_fraction = target_type_fraction
        self.phases = np.asarray(phases) if phases is not None else phases
        self.orientations = orientations

        self._material = None

        if self.phases is not None and self.target_type_fraction is not None:
            raise ValueError("Cannot specify both `phases` and `target_type_fraction`.")

        if orientations is not None and phases is None:
            raise ValueError(
                "If specifying `orientations`, must also specify `phases`."
            )

    @property
    def material(self):
        return self._material

    @property
    def name(self):
        return self.material.name + (f"-{self.type_label}" if self.type_label else "")


class CIPHERGeometry:
    def __init__(
        self,
        materials,
        interfaces,
        size,
        seeds=None,
        voxel_phase=None,
        voxel_map=None,
        random_seed=None,
    ):

        if sum(i is not None for i in (voxel_phase, voxel_map)) != 1:
            raise ValueError(f"Specify exactly one of `voxel_phase` and `voxel_map`")
        if voxel_map is None:
            voxel_map = VoxelMap(region_ID=voxel_phase, size=size, is_periodic=True)
        else:
            voxel_phase = voxel_map.region_ID

        self._interfaces = None

        self.voxel_map = voxel_map
        self.voxel_phase = voxel_phase
        self.seeds = seeds
        self.materials = materials
        self.interfaces = interfaces
        self.size = np.asarray(size)

        for i in self.materials:
            i._geometry = self

        for idx, i in enumerate(self.interfaces):
            i.index = idx

        if self.size.size != self.dimension:
            raise ValueError(
                f"`size` ({self.size}) implies {self.size.size} dimensions, but "
                f"`voxel_phase` implies {self.voxel_phase.dimension} dimensions."
            )

        all_phases = np.unique(self.voxel_phase)
        self._num_phases = all_phases.size

        if not np.all(all_phases == np.arange(self.num_phases)):
            raise ValueError(
                "`voxel_phase` must be an array of consecutive integers starting from "
                "zero."
            )
            # TODO: test raise

        if len(set(self.material_names)) < self.num_materials:
            raise ValueError(
                f"Repeated material names exist in the materials definitions: "
                f"{self.material_names!r}."
            )

        self._ensure_phase_assignment(random_seed)

        self._phase_material = self._get_phase_material()
        self._validate_interfaces()
        self._check_interface_phase_pairs()

        self._phase_phase_type = self._get_phase_phase_type()
        self._phase_num_voxels = self._get_phase_num_voxels()
        self._interface_map = self._get_interface_map()
        self._validate_interface_map()  # TODO: add setter to interface map

        self._phase_orientation = self._get_phase_orientation()

    def _validate_interfaces(self):
        int_names = self.interface_names
        if len(set(int_names)) < len(int_names):
            raise ValueError(
                f"Multiple interfaces have the same name (i.e. "
                f"phase-type-pair and type-label combination)!"
            )

    @property
    def interfaces(self):
        return self._interfaces

    @interfaces.setter
    def interfaces(self, interfaces):
        self._interfaces = interfaces
        self._validate_interfaces()

    def _get_phase_num_voxels(self):
        return np.array(
            [
                np.sum(self.voxel_phase == phase_idx)
                for phase_idx in range(self.num_phases)
            ]
        )

    def _ensure_phase_assignment(self, random_seed):
        is_mat_phases = [i.phases is not None for i in self.materials]
        is_mat_vol_frac = [i is not None for i in self.target_material_volume_fractions]
        is_mixed = any(is_mat_phases) and any(is_mat_vol_frac)

        if is_mixed or (any(is_mat_phases) and not all(is_mat_phases)):
            raise ValueError(
                f"Specify either: all phases explicitly (via the material definition "
                f"`phases`, or the constituent phase type definition `phases`), or "
                f"specify zero or more target volume fractions for the material "
                f"definitions."
            )  # TODO: test raise

        if not any(is_mat_phases):
            self._assign_phases_by_volume_fractions(is_mat_vol_frac, random_seed)

    def _check_interface_phase_pairs(self):
        """If interfaces have phase-pairs specified, check these are consistent with
        the specified phases of associated material."""

        for i in self.interfaces:
            if i.phase_pairs.size:
                mats_idx = np.sort([self.material_names.index(j) for j in i.materials])
                phase_pairs_material = self.phase_material[i.phase_pairs]
                phase_pairs_mat_srt = np.sort(phase_pairs_material, axis=1)
                if not np.all(np.all(phase_pairs_mat_srt == mats_idx, axis=1)):
                    raise ValueError(
                        f"Phase pairs specified for interface {i.name!r} are not "
                        f"consistent with phases specified for the interface materials "
                        f"{i.materials[0]!r} and {i.materials[1]!r}."
                    )  # TODO: test raise

    def _assign_phases_by_volume_fractions(self, is_mat_vol_frac, random_seed):
        # Assign via target volume fractions.
        num_unassigned_vol = self.num_materials - sum(is_mat_vol_frac)
        assigned_vol = sum(i or 0.0 for i in self.target_material_volume_fractions)
        if num_unassigned_vol:
            frac = (1.0 - assigned_vol) / num_unassigned_vol
            if frac <= 0.0:
                raise ValueError(
                    f"All material target volume fractions must sum to one, but "
                    f"assigned target volume fractions sum to {assigned_vol} with "
                    f"{num_unassigned_vol} outstanding unassigned material volume "
                    f"fraction(s)."
                )  # TODO: test raise
        for i in self.materials:
            if i.target_volume_fraction is None:
                i.target_volume_fraction = frac

        assigned_vol = sum(self.target_material_volume_fractions)
        if not np.isclose(assigned_vol, 1.0):
            raise ValueError(
                f"All material target volume fractions must sum to one, but target "
                f"volume fractions sum to {assigned_vol}."
            )  # TODO: test raise

        # Now assign phases:
        rng = np.random.default_rng(seed=random_seed)
        phase_material = rng.choice(
            a=self.num_materials,
            size=self.num_phases,
            p=self.target_material_volume_fractions,
        )
        for mat_idx, mat in enumerate(self.materials):
            mat_phases = np.where(phase_material == mat_idx)[0]
            mat.assign_phases(mat_phases)

    def _get_phase_material(self):
        phase_material = np.ones(self.num_phases) * np.nan
        for mat_idx, mat in enumerate(self.materials):
            phase_material[mat.phases] = mat_idx
        if np.any(np.isnan(phase_material)):
            raise ValueError(
                "Not all phases are accounted for in the phase type definitions."
            )  # TODO: test raise
        return phase_material.astype(int)

    def _get_phase_phase_type(self):
        phase_phase_type = np.ones(self.num_phases) * np.nan
        for phase_type_idx, phase_type in enumerate(self.phase_types):
            phase_phase_type[phase_type.phases] = phase_type_idx
        if np.any(np.isnan(phase_phase_type)):
            raise RuntimeError("Not all phases accounted for!")  # TODO: test raise?
        return phase_phase_type.astype(int)

    def _get_phase_orientation(self):
        """Get the orientation of each phase, if specified in the phase-type."""
        phase_ori = np.ones((self.num_phases, 4), dtype=float) * np.nan
        for phase_type in self.phase_types:
            for type_idx, phase_i in enumerate(phase_type.phases):
                if phase_type.orientations is not None:
                    phase_ori[phase_i] = phase_type.orientations[type_idx]
        return phase_ori

    def get_interface_map_indices(self, phase_type_A, phase_type_B):
        """Get an array of integer indices that index the (upper triangle of the) 2D
        symmetric interface map array, corresponding to a given material pair."""

        # First get phase indices belonging to the two phase types:
        ptypes = {i.name: i for i in self.phase_types}
        ptA_phases = ptypes[phase_type_A].phases
        ptB_phases = ptypes[phase_type_B].phases

        A_idx = np.repeat(ptA_phases, ptB_phases.shape[0])
        B_idx = np.tile(ptB_phases, ptA_phases.shape[0])

        map_idx = np.vstack((A_idx, B_idx))
        map_idx_srt = np.sort(map_idx, axis=0)  # map onto upper triangle
        map_idx_uniq = np.unique(map_idx_srt, axis=1)  # get unique pairs only

        # remove diagonal elements (a phase can't have an interface with itself)
        map_idx_non_trivial = map_idx_uniq[:, map_idx_uniq[0] != map_idx_uniq[1]]

        return map_idx_non_trivial

    def _get_interface_map(self, upper_tri_only=False):
        """Generate the num_phases by num_phases symmetric matrix that maps each phase-pair
        to an interface index."""

        print("Finding interface map matrix...", end="")

        int_map = np.ones((self.num_phases, self.num_phases), dtype=int) * np.nan

        ints_by_phase_type_pair = {}
        for int_def in self.interfaces:
            if int_def.phase_types not in ints_by_phase_type_pair:
                ints_by_phase_type_pair[int_def.phase_types] = []
            ints_by_phase_type_pair[int_def.phase_types].append(int_def)

        for pt_pair, int_defs in ints_by_phase_type_pair.items():

            # print(f"int_defs: {int_defs}")

            names = [i.name for i in int_defs]
            if len(set(names)) < len(names):
                raise ValueError(
                    f"Multiple interface definitions for phase-type pair "
                    f"{pt_pair} have the same `type_label`."
                )
            type_fracs = [i.type_fraction for i in int_defs]
            any_frac_set = any(i is not None for i in type_fracs)
            manual_set = [i.is_phase_pairs_set for i in int_defs]
            any_manual_set = any(manual_set)
            all_manual_set = all(manual_set)
            if any_frac_set:
                if any_manual_set:
                    raise ValueError(
                        f"For interface {pt_pair}, specify phase pairs manually for all "
                        f"defined interfaces using `phase_pairs`, or specify `type_fraction`"
                        f"for all defined interfaces. You cannot mix them."
                    )

            all_phase_pairs = self.get_interface_map_indices(*pt_pair).T
            if any_manual_set:
                if not all_manual_set:
                    raise ValueError(
                        f"For interface {pt_pair}, specify phase pairs manually for all "
                        f"defined interfaces using `phase_pairs`, or specify `type_fraction`"
                        f"for all defined interfaces. You cannot mix them."
                    )

                # check that given phase_pairs combine to the set of all phase_pairs
                # for this material-material pair:
                all_given_phase_pairs = np.vstack([i.phase_pairs for i in int_defs])

                # sort by first-phase, then second-phase, for comparison:
                srt = np.lexsort(all_given_phase_pairs.T[::-1])
                all_given_phase_pairs = all_given_phase_pairs[srt]

                if all_given_phase_pairs.shape != all_phase_pairs.shape or not np.all(
                    all_given_phase_pairs == all_phase_pairs
                ):
                    raise ValueError(
                        f"Missing `phase_pairs` for interface {pt_pair}. The following "
                        f"phase pairs must all be included for this interface: "
                        f"{all_phase_pairs}"
                    )

                for int_i in int_defs:
                    phase_pairs_i = int_i.phase_pairs.T
                    if phase_pairs_i.size:
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
                            f"For interface {pt_pair}, `type_fraction` for all "
                            f"defined interfaces must sum to one."
                        )
                    remainder_frac_each = remainder_frac / num_missing_type_frac
                    for i in int_defs:
                        if i.type_fraction is None:
                            i.type_fraction = remainder_frac_each

                type_fracs = [i.type_fraction for i in int_defs]
                if sum(type_fracs) != 1:
                    raise ValueError(
                        f"For interface {pt_pair}, `type_fraction` for all "
                        f"defined interfaces must sum to one."
                    )

                # assign phase_pairs according to type fractions:
                num_pairs = all_phase_pairs.shape[0]
                type_nums_each = [round(i * num_pairs) for i in type_fracs]
                type_nums = np.cumsum(type_nums_each)
                if num_pairs % 2 == 1:
                    type_nums += 1

                shuffle_idx = np.random.choice(num_pairs, size=num_pairs, replace=False)
                phase_pairs_shuffled = all_phase_pairs[shuffle_idx]
                phase_pairs_split = np.split(phase_pairs_shuffled, type_nums, axis=0)[
                    :-1
                ]
                for idx, int_i in enumerate(int_defs):
                    phase_pairs_i = phase_pairs_split[idx]
                    int_map[phase_pairs_i[:, 0], phase_pairs_i[:, 1]] = int_i.index
                    if not upper_tri_only:
                        int_map[phase_pairs_i[:, 1], phase_pairs_i[:, 0]] = int_i.index

        print("done!")

        return int_map

    @property
    def interface_map_int(self):
        """Get the interface map as an integer matrix, where NaNs are replaced by -2."""
        int_map = np.copy(self.interface_map)
        int_map[np.isnan(int_map)] = -2
        return int_map.astype(int)

    def get_interface_idx(self):
        return self.voxel_map.get_interface_idx(self.interface_map_int)

    def _modify_interface_map(self, phase_A, phase_B, interface_idx):
        """
        Parameters
        ----------
        phase_A : ndarray
        phase_B : ndarray
        interface_idx : ndarray

        """
        if interface_idx not in range(len(self.interfaces)):
            raise ValueError(f"Interface index {interface_idx} invalid.")
        self._interface_map[phase_A, phase_B] = interface_idx
        self._interface_map[phase_B, phase_A] = interface_idx

    def _validate_interface_map(self):
        # check no missing interfaces:
        int_map_indices = np.triu_indices_from(self.interface_map, k=1)
        int_is_nan = np.isnan(self.interface_map[int_map_indices])
        phase_idx_int_is_nan = np.vstack(int_map_indices)[:, int_is_nan]
        if phase_idx_int_is_nan.size:
            raise RuntimeError(
                f"The following phase-pairs have not been assigned an interface "
                f"definition: {phase_idx_int_is_nan}."
            )

    def get_misorientation_matrix(self):
        """Given phase type definitions that include orientation lists, get the
        misorientation matrix between all pairs."""

        misori_matrix = np.zeros((self.num_phases, self.num_phases), dtype=float)
        all_oris = []
        for i in self.phase_types:
            all_oris.extend(i.orientations)
        all_oris = np.vstack(all_oris)

        for idx_i, ori_i in enumerate(all_oris):
            other_oris = all_oris[idx_i + 1 :]
            # TODO: consider symmetry!
            misoris_i = quat_angle_between(
                np.tile(ori_i, (other_oris.shape[0], 1)),
                other_oris,
            )
            misori_matrix[idx_i, idx_i + 1 :] = misoris_i
            misori_matrix[idx_i + 1 :, idx_i] = misoris_i

        return misori_matrix

    def get_pyvista_grid(self):
        """Experimental!"""

        grid = pv.UniformGrid()

        grid.dimensions = self.grid_size_3D + 1  # +1 to inject values on cell data
        grid.spacing = self.size_3D / self.grid_size_3D
        return grid

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
        interfaces,
        materials,
        grid_size,
        size,
        seeds=None,
        num_phases=None,
        random_seed=None,
    ):

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

        return cls(
            voxel_map=vor_map,
            materials=materials,
            interfaces=interfaces,
            size=size,
            seeds=seeds,
            random_seed=random_seed,
        )

    @classmethod
    def from_seed_voronoi(
        cls,
        seeds,
        interfaces,
        materials,
        grid_size,
        size,
        random_seed=None,
    ):
        return cls.from_voronoi(
            interfaces=interfaces,
            materials=materials,
            grid_size=grid_size,
            size=size,
            seeds=seeds,
            random_seed=random_seed,
        )

    @classmethod
    def from_random_voronoi(
        cls,
        num_phases,
        interfaces,
        materials,
        grid_size,
        size,
        random_seed=None,
    ):
        return cls.from_voronoi(
            interfaces=interfaces,
            materials=materials,
            grid_size=grid_size,
            size=size,
            num_phases=num_phases,
            random_seed=random_seed,
        )

    @property
    def voxel_phase_3D(self):
        if self.dimension == 3:
            return self.voxel_phase
        else:
            return self.voxel_phase.T[:, :, None]

    @property
    def voxel_material_3D(self):
        if self.dimension == 3:
            return self.voxel_material
        else:
            return self.voxel_material.T[:, :, None]

    @property
    def voxel_interface_idx_3D(self):
        int_idx = self.get_interface_idx()
        if self.dimension == 3:
            return int_idx
        else:
            return int_idx.T[:, :, None]

    def show(self):
        """Experimental!"""

        print("WARNING: experimental!")

        grid = self.get_pyvista_grid()

        grid.cell_data["interface_idx"] = self.voxel_interface_idx_3D.flatten(order="F")
        grid.cell_data["material"] = self.voxel_material_3D.flatten(order="F")
        grid.cell_data["phase"] = self.voxel_phase_3D.flatten(order="F")

        pl = pv.PlotterITK()
        pl.add_mesh(grid)
        pl.show(ui_collapsed=False)

    @property
    def dimension(self):
        return self.voxel_map.dimension

    @property
    def grid_size(self):
        return np.array(self.voxel_map.grid_size)

    @property
    def grid_size_3D(self):
        if self.dimension == 2:
            return np.hstack([self.grid_size[::-1], 1])
        else:
            return self.grid_size

    @property
    def size_3D(self):
        if self.dimension == 2:
            return np.hstack([self.size[::-1], self.size[0] / self.grid_size[0]])
        else:
            return self.size

    @property
    def neighbour_voxels(self):
        return self.voxel_map.neighbour_voxels

    @property
    def neighbour_list(self):
        return self.voxel_map.neighbour_list

    @property
    def interface_map(self):
        """Get the num_phases-by-num_phases matrix of interface indices."""
        return self._interface_map

    @property
    def interface_names(self):
        return [i.name for i in self.interfaces]

    @property
    def material_properties(self):
        return {mat.name: mat.properties for mat in self.materials}

    @property
    def num_voxels(self):
        return np.product(
            self.voxel_phase.size
        )  # TODO: change to voxel_map.num_voxels?

    @property
    def phase_num_voxels(self):
        return self._phase_num_voxels

    @property
    def num_phases(self):
        return self._num_phases

    @property
    def num_materials(self):
        return len(self.materials)

    @property
    def material_names(self):
        return [i.name for i in self.materials]

    @property
    def target_material_volume_fractions(self):
        return [i.target_volume_fraction for i in self.materials]

    @property
    def phase_types(self):
        """Get all phase types across all materials."""
        return [j for i in self.materials for j in i.phase_types]

    @property
    def phase_material(self):
        """Get the material index of each phase."""
        return self._phase_material

    @property
    def phase_phase_type(self):
        """Get the phase type index of each phase."""
        return self._phase_phase_type

    @property
    def phase_orientation(self):
        """Get the orientation quaternion of each phase."""
        return self._phase_orientation

    @property
    def voxel_material(self):
        """Get the material index of each voxel."""
        return self.phase_material[self.voxel_phase]

    @property
    def voxel_phase_type(self):
        """Get the phase type index of each voxel."""
        return self.phase_phase_type[self.voxel_phase]

    @property
    def voxel_orientation(self):
        """Get the quaternion of each voxel."""
        return self.phase_orientation[self.voxel_phase]

    @property
    def material_num_voxels(self):
        mat_num_vox = []
        for i in self.materials:
            num_vox_i = sum(self.phase_num_voxels[j] for j in i.phases)
            mat_num_vox.append(num_vox_i)
        return np.array(mat_num_vox)

    @property
    def phase_type_num_voxels(self):
        return np.array(
            [np.sum(self.phase_num_voxels[i.phases]) for i in self.phase_types]
        )

    @property
    def material_volume_fractions(self):
        return np.array([i / self.num_voxels for i in self.material_num_voxels])

    @property
    def phase_volume_fractions(self):
        return np.array([i / self.num_voxels for i in self.phase_num_voxels])

    @property
    def phase_type_volume_fractions(self):
        return np.array([i / self.num_voxels for i in self.phase_type_num_voxels])

    @property
    def seeds_grid(self):
        return np.round(self.grid_size * self.seeds / self.size, decimals=0).astype(int)


@dataclass
class CIPHERInput:
    geometry: CIPHERGeometry
    components: List
    outputs: List
    solution_parameters: Dict

    def __post_init__(self):
        self._validate()

    def _validate(self):
        check_grid_size = (
            np.array(self.solution_parameters["initblocksize"])
            * 2 ** self.solution_parameters["initrefine"]
        )
        if not np.all(check_grid_size == np.array(self.geometry.grid_size)):
            raise ValueError(
                f"`grid_size` (specifed: {self.geometry.grid_size}) must be equal to: "
                f"`initblocksize` (specified: {self.solution_parameters['initblocksize']}) "
                f"multiplied by 2 raised to the power of `initrefine` (specified: "
                f"{self.solution_parameters['initrefine']}), calculated to be: "
                f"{check_grid_size}."
            )

    @classmethod
    def from_voronoi(
        cls,
        grid_size,
        size,
        materials,
        interfaces,
        components,
        outputs,
        solution_parameters,
        seeds=None,
        num_phases=None,
        random_seed=None,
    ):

        geometry = CIPHERGeometry.from_voronoi(
            num_phases=num_phases,
            seeds=seeds,
            interfaces=interfaces,
            materials=materials,
            grid_size=grid_size,
            size=size,
            random_seed=random_seed,
        )

        inp = cls(
            geometry=geometry,
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
    ):

        geometry = CIPHERGeometry(
            voxel_phase=voxel_phase,
            materials=materials,
            interfaces=interfaces,
            size=size,
            random_seed=random_seed,
        )
        inp = cls(
            geometry=geometry,
            components=components,
            outputs=outputs,
            solution_parameters=solution_parameters,
        )
        return inp

    @classmethod
    def from_dream3D(
        cls,
        path,
        materials,
        interfaces,
        components,
        outputs,
        solution_parameters,
        container_labels=None,
        phase_type_map=None,
    ):

        default_container_labels = {
            "SyntheticVolumeDataContainer": "SyntheticVolumeDataContainer",
            "CellData": "CellData",
            "CellEnsembleData": "CellEnsembleData",
            "FeatureIds": "FeatureIds",
            "Grain Data": "Grain Data",
            "Phases": "Phases",
            "NumFeatures": "NumFeatures",
            "BoundaryCells": "BoundaryCells",
            "NumNeighbors": "NumNeighbors",
            "NeighborList": "NeighborList",
            "SharedSurfaceAreaList": "SharedSurfaceAreaList",
            "SurfaceFeatures": "SurfaceFeatures",
            "AvgQuats": "AvgQuats",
        }
        container_labels = container_labels or {}
        container_labels = {**default_container_labels, **container_labels}

        with h5py.File(path, "r") as fp:

            voxel_phase_path = "/".join(
                (
                    "DataContainers",
                    container_labels["SyntheticVolumeDataContainer"],
                    container_labels["CellData"],
                    container_labels["FeatureIds"],
                )
            )
            phase_material_path = "/".join(
                (
                    "DataContainers",
                    container_labels["SyntheticVolumeDataContainer"],
                    container_labels["Grain Data"],
                    container_labels["Phases"],
                )
            )
            spacing_path = "/".join(
                (
                    "DataContainers",
                    container_labels["SyntheticVolumeDataContainer"],
                    "_SIMPL_GEOMETRY",
                    "SPACING",
                )
            )
            dims_path = "/".join(
                (
                    "DataContainers",
                    container_labels["SyntheticVolumeDataContainer"],
                    "_SIMPL_GEOMETRY",
                    "DIMENSIONS",
                )
            )
            material_names_path = "/".join(
                (
                    "DataContainers",
                    container_labels["SyntheticVolumeDataContainer"],
                    container_labels["CellEnsembleData"],
                    "PhaseName",
                )
            )
            grain_quats_path = "/".join(
                (
                    "DataContainers",
                    container_labels["SyntheticVolumeDataContainer"],
                    container_labels["Grain Data"],
                    container_labels["AvgQuats"],
                )
            )

            voxel_phase = fp[voxel_phase_path][()][:, :, :, 0]
            phase_material = fp[phase_material_path][()].flatten()
            voxel_phase = np.transpose(voxel_phase, axes=[2, 1, 0])
            spacing = fp[spacing_path][()]  # same as "resolution" in GUI
            dimensions = fp[dims_path][()]
            size = np.array([i * j for i, j in zip(spacing, dimensions)])
            mat_names = [i.decode("utf-8") for i in fp[material_names_path]]
            grain_quats = fp[grain_quats_path][()]

        # ignore unknown phase:
        phase_material = phase_material[1:] - 1
        voxel_phase = voxel_phase - 1
        mat_names = mat_names[1:]

        for mat_idx, mat_name_i in enumerate(mat_names):
            phases_set = False
            if phase_type_map:
                phase_type_name = phase_type_map[mat_name_i]
            else:
                phase_type_name = mat_name_i
            for mat in materials:
                for phase_type_i in mat.phase_types:
                    if phase_type_i.name == phase_type_name:
                        phase_i_idx = np.where(phase_material == mat_idx)[0]
                        phase_type_i.phases = phase_i_idx
                        phase_type_i.orientations = grain_quats[phase_i_idx]
                        phases_set = True
                        break
                if phases_set:
                    break

            if not phases_set:
                raise ValueError(
                    f"No defined material/phase-type for Dream3D phase {mat_name_i!r}"
                )

        return cls.from_voxel_phase_map(
            voxel_phase=voxel_phase,
            size=size,
            materials=materials,
            interfaces=interfaces,
            components=components,
            outputs=outputs,
            solution_parameters=solution_parameters,
        )

    @property
    def materials(self):
        return self.geometry.materials

    @property
    def material_properties(self):
        return self.geometry.material_properties

    @property
    def interfaces(self):
        return self.geometry.interfaces

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
            "material": {
                k: copy.deepcopy(v) for k, v in self.material_properties.items()
            },
            "interface": {
                k: copy.deepcopy(v) for k, v in self.get_interfaces().items()
            },
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
                    compress_1D_array_string(
                        self.geometry.interface_map_int.flatten() + 1
                    )
                    + "\n"
                ),
            },
        }

        yaml = YAML()
        path = Path(path)
        with path.open("wt", newline="\n") as fp:
            yaml.dump(cipher_input_data, fp)

        return path

    def apply_interface_property(
        self,
        base_interface_name,
        property_name,
        property_values,
        additional_metadata=None,
        bin_edges=None,
    ):
        """Expand a base interface into multiple interfaces, by assigning the specified
        property (e.g. GB energy) from a symmetric matrix of property values

        Parameters
        ----------
        base_interface_name : str
        property : tuple of str
        property_values : ndarray of shape (N_phases, N_phases)
            N_phases it the total number of phases in the geometry.
        bin_edges : ndarray of float, optional
            If specified, bin property values such that multiple phase-pairs are
            represented by the same interface definition. This uses `np.digitize`.

        """

        base_interface_idx = self.geometry.interface_names.index(base_interface_name)
        base_interface_defn = self.geometry.interfaces.pop(base_interface_idx)
        interface_map_tri = np.tril(
            -np.ones_like(self.geometry.interface_map)
        ) + np.triu(self.geometry.interface_map)
        phase_pairs = np.array(np.where(interface_map_tri == base_interface_idx))

        new_vals_all = property_values[phase_pairs[0], phase_pairs[1]]
        new_interfaces_data = []
        if bin_edges is not None:
            bin_idx = np.digitize(new_vals_all, bin_edges)
            all_pp_idx_i = []
            for idx, bin_i in enumerate(bin_edges):
                pp_idx_i = np.where(bin_idx == idx + 1)[0]
                all_pp_idx_i.extend(pp_idx_i.tolist())
                if pp_idx_i.size:
                    print(
                        f"Adding {pp_idx_i.size!r} phase pair(s) to {property_name!r} bin "
                        f"{idx + 1} with edge value: {bin_i!r}"
                    )
                    new_interfaces_data.append(
                        {
                            "phase_pairs": phase_pairs.T[pp_idx_i],
                            "value": bin_i,
                        }
                    )

            miss_phase_pairs = set(np.arange(phase_pairs.shape[1])) - set(all_pp_idx_i)
            if miss_phase_pairs:
                miss_prop_vals = [
                    property_values[phase_pairs[0, i], phase_pairs[1, i]]
                    for i in miss_phase_pairs
                ]
                missing_dat = dict(
                    zip(
                        (tuple(phase_pairs[:, i]) for i in miss_phase_pairs),
                        miss_prop_vals,
                    )
                )
                raise RuntimeError(
                    f"Not all phase pairs have been added to a property value bin. The "
                    f"following {len(missing_dat)}/{phase_pairs.shape[1]} phase pairs (and "
                    f"property values) are missing: {missing_dat}."
                )
        else:
            print(
                f"Adding a new interface for each of {phase_pairs.shape[1]} phase pairs."
            )
            new_interfaces_data = [
                {
                    "phase_pairs": np.array([pp]),
                    "value": new_vals_all[pp_idx],
                }
                for pp_idx, pp in enumerate(phase_pairs.T)
            ]

        num_existing_int_defns = len(self.geometry.interfaces)
        print("Preparing new interface defintions...", end="")
        for idx, i in enumerate(new_interfaces_data):

            props = copy.deepcopy(base_interface_defn.properties)
            new_value = i["value"].item()  #  convert from numpy to native
            set_by_path(root=props, path=property_name, value=new_value)

            new_type_lab = str(idx)
            if base_interface_defn.type_label:
                new_type_lab = f"{base_interface_defn.type_label}-{new_type_lab}"

            metadata = {}
            if additional_metadata:
                for k, v in additional_metadata.items():
                    metadata[k] = v[i["phase_pairs"][:, 0], i["phase_pairs"][:, 1]]

            new_int = InterfaceDefinition(
                phase_types=base_interface_defn.phase_types,
                type_label=new_type_lab,
                properties=props,
                phase_pairs=i["phase_pairs"].tolist(),
                metadata=metadata,
            )
            self.geometry.interfaces.append(new_int)
            self.geometry._modify_interface_map(
                phase_A=i["phase_pairs"][:, 0],
                phase_B=i["phase_pairs"][:, 1],
                interface_idx=(num_existing_int_defns + idx),
            )

        print("done!")
        self.geometry._validate_interfaces()
        self.geometry._validate_interface_map()
