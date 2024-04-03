import os
import math
import logging
import logging.handlers
import subprocess
import pickle
from pathlib import Path

import numpy as np

from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cssr import Cssr

from ase.io import read
from ase.neighborlist import natural_cutoffs
from ase import neighborlist
from ase.build import make_supercell


cur_dir = Path(__file__).parent

GRIDAY_PATH = os.path.join(cur_dir, "libs/GRIDAYS/scripts/grid_gen")
FF_PATH = os.path.join(cur_dir, "libs/GRIDAYS/FF")


def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_unique_atoms(atoms):
    # get graph
    cutoff = natural_cutoffs(atoms)
    neighbor_list = neighborlist.NeighborList(
        cutoff, self_interaction=True, bothways=True
    )
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix()

    # Get N, N^2
    numbers = atoms.numbers
    number_sqr = np.multiply(numbers, numbers)

    matrix_sqr = matrix.dot(matrix)
    matrix_cub = matrix_sqr.dot(matrix)
    matrix_sqr.data[:] = 1  # count 1 for atoms

    # calculate
    list_n = [numbers, number_sqr]
    list_m = [matrix, matrix_sqr, matrix_cub]

    arr = [numbers]

    for m in list_m:
        for n in list_n:
            arr.append(m.dot(n))

    arr = np.vstack(arr).transpose()

    uni, unique_idx, unique_count = np.unique(
        arr, axis=0, return_index=True, return_counts=True
    )

    # sort
    final_uni = uni[np.argsort(-unique_count)].tolist()
    final_unique_count = unique_count[np.argsort(-unique_count)].tolist()

    arr = arr.tolist()
    final_unique_idx = []
    for u in final_uni:
        final_unique_idx.append([i for i, a in enumerate(arr) if a == u])

    return final_unique_idx, final_unique_count


def get_crystal_graph(atoms, radius=8, max_num_nbr=12):
    dist_mat = atoms.get_all_distances(mic=True)
    nbr_mat = np.where(dist_mat > 0, dist_mat, 1000)  # 1000 is mamium number
    nbr_idx = []
    nbr_dist = []
    for row in nbr_mat:
        idx = np.argsort(row)[:max_num_nbr]
        nbr_idx.extend(idx)
        nbr_dist.extend(row[idx])

    # get same-topo atoms
    uni_idx, uni_count = get_unique_atoms(atoms)

    # convert to small size
    atom_num = np.array(list(atoms.numbers), dtype=np.int8)
    nbr_idx = np.array(nbr_idx, dtype=np.int16)
    nbr_dist = np.array(nbr_dist, dtype=np.float32)
    uni_count = np.array(uni_count, dtype=np.int16)
    return atom_num, nbr_idx, nbr_dist, uni_idx, uni_count


def _calculate_scaling_matrix_for_orthogonal_supercell(cell_matrix, eps=0.01):
    """
    cell_matrix: contains lattice vector as column vectors.
                 e.g. cell_matrix[:, 0] = a.
    eps: when value < eps, the value is assumed as zero.
    """
    inv = np.linalg.inv(cell_matrix)

    # Get minimum absolute values of each row.
    abs_inv = np.abs(inv)
    mat = np.where(abs_inv < eps, np.full_like(abs_inv, 1e30), abs_inv)
    min_values = np.min(mat, axis=1)

    # Normalize each row with minimum absolute value of each row.
    normed_inv = inv / min_values[:, np.newaxis]

    # Calculate scaling_matrix.
    # New cell = np.dot(scaling_matrix, cell_matrix).
    scaling_matrix = np.around(normed_inv).astype(np.int32)

    return scaling_matrix


def get_energy_grid(atoms, cif_id, root_dataset, eg_logger):
    # Before 1.1.1 version : num_grid = [str(round(cell)) for cell in structure.lattice.abc]
    # After 1.1.1 version : num_grid = [30, 30, 30]
    global GRIDAY_PATH, FF_PATH

    eg_file = os.path.join(root_dataset, cif_id)
    # random_str = str(np.random.rand()).encode()
    tmp_file = os.path.join(root_dataset, f"{cif_id}.cssr")

    try:
        structure = AseAtomsAdaptor().get_structure(atoms)
        Cssr(structure).write_file(tmp_file)
        if not os.path.exists(tmp_file):
            eg_logger.info(f"{cif_id} cssr write fail")
            return False
        num_grid = ["30", "30", "30"]
        proc = subprocess.Popen(
            [
                GRIDAY_PATH,
                *num_grid,
                f"{FF_PATH}/UFF_Type.def",
                f"{FF_PATH}/UFF_FF.def",
                tmp_file,
                eg_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()
    finally:
        # remove temp_file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    if err:
        eg_logger.info(f"{cif_id} energy grid failed {err}")
        return False
    else:
        eg_logger.info(f"{cif_id} energy grid success")

    return True


def _make_supercell(atoms, cutoff):
    """
    make atoms into supercell when cell length is less than cufoff (min_length)
    """
    # when the cell lengths are smaller than radius, make supercell to be longer than the radius
    scale_abc = []
    for l in atoms.cell.cellpar()[:3]:
        if l < cutoff:
            scale_abc.append(math.ceil(cutoff / l))
        else:
            scale_abc.append(1)

    # make supercell
    m = np.zeros([3, 3])
    np.fill_diagonal(m, scale_abc)
    atoms = make_supercell(atoms, m)
    return atoms


def make_prepared_data(
    cif: Path, root_dataset_total: Path, logger=None, eg_logger=None, **kwargs
):
    if logger is None:
        logger = get_logger(filename="prepare_data.log")
    if eg_logger is None:
        eg_logger = get_logger(filename="prepare_energy_grid.log")

    if isinstance(cif, str):
        cif = Path(cif)
    if isinstance(root_dataset_total, str):
        root_dataset_total = Path(root_dataset_total)

    root_dataset_total.mkdir(exist_ok=True, parents=True)

    max_length = kwargs.get("max_length", 60.0)
    min_length = kwargs.get("min_length", 30.0)
    max_num_nbr = kwargs.get("max_num_nbr", 12)
    max_num_unique_atoms = kwargs.get("max_num_unique_atoms", 300)
    max_num_atoms = kwargs.get("max_num_atoms", None)

    cif_id: str = cif.stem

    p_graphdata = root_dataset_total / f"{cif_id}.graphdata"
    p_griddata = root_dataset_total / f"{cif_id}.griddata16"
    p_grid = root_dataset_total / f"{cif_id}.grid"

    # # Grid data and Graph data already exists
    # if p_graphdata.exists() and p_griddata.exists() and p_grid.exists():
    #     logger.info(f"{cif_id} graph data already exists")
    #     eg_logger.info(f"{cif_id} energy grid already exists")
    #     return True

    # valid cif check
    try:
        CifParser(cif).get_structures()
    except ValueError as e:
        logger.info(f"{cif_id} failed : {e} (error when reading cif with pymatgen)")
        return False

    # read cif by ASE
    try:
        atoms = read(str(cif))
    except Exception as e:
        logger.error(f"{cif_id} failed : {e}")
        return False

    # 1. get crystal graph
    atoms = _make_supercell(atoms, cutoff=8)  # radius = 8

    if max_num_atoms and len(atoms) > max_num_atoms:
        logger.error(
            f"{cif_id} failed : number of atoms are larger than `max_num_atoms` ({max_num_atoms})"
        )
        return False

    atom_num, nbr_idx, nbr_dist, uni_idx, uni_count = get_crystal_graph(
        atoms, radius=8, max_num_nbr=max_num_nbr
    )
    if len(nbr_idx) < len(atom_num) * max_num_nbr:
        logger.error(
            f"{cif_id} failed : num_nbr is smaller than max_num_nbr. please make radius larger"
        )
        return False

    if len(uni_idx) > max_num_unique_atoms:
        logger.error(
            f"{cif_id} failed : The number of topologically unique atoms is larget than `max_num_unique_atoms` ({max_num_unique_atoms})"
        )
        return False

    # 2. make supercell with min_length
    atoms_eg = _make_supercell(atoms, cutoff=min_length)
    for l in atoms_eg.cell.cellpar()[:3]:
        if l > max_length:
            logger.error(f"{cif_id} failed : supercell have more than max_length")
            return False

    # 3. calculate energy grid
    eg_success = get_energy_grid(atoms_eg, cif_id, root_dataset_total, eg_logger)

    if eg_success:
        logger.info(f"{cif_id} succeed : supercell length {atoms.cell.cellpar()[:3]}")

        # save cif files
        save_cif_path = root_dataset_total / f"{cif_id}.cif"
        atoms.write(filename=save_cif_path)

        # save graphdata file
        data = [cif_id, atom_num, nbr_idx, nbr_dist, uni_idx, uni_count]
        with open(str(p_graphdata), "wb") as f:
            pickle.dump(data, f)
        return True
    else:
        return False
