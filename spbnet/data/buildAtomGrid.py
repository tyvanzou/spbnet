from pathlib import Path
import math
import numpy as np
from .prepare_data import _make_supercell
from ase.io import read
import argparse
from spbnet.utils.echo import title, param, err
from multiprocessing import Process
from tqdm import tqdm


GRID = 30


def get_atoms(cif_path: Path):
    atoms = read(cif_path.absolute())
    atoms = _make_supercell(atoms, cutoff=30)
    return atoms


def cif2array(cif_path: Path):
    ret = np.zeros((GRID, GRID, GRID))

    atoms = get_atoms(
        cif_path=cif_path,
    )
    frac_positions = atoms.get_scaled_positions()
    atomic_numbers = atoms.get_atomic_numbers()

    # 计算小格子中每个原子的个数
    for frac_coords, _ in zip(frac_positions, atomic_numbers):
        frac_ints = [math.floor(frac_coords[i] * GRID) for i in range(3)]
        ret[frac_ints[0], frac_ints[1], frac_ints[2]] += 1

    return ret


def buildAtomGridi(cif_path: Path, tgt_dir: Path):
    (tgt_dir / "atomgrid").mkdir(exist_ok=True)
    array = cif2array(cif_path)  # [GRID, GRID, GRID]
    np.save((tgt_dir / "atomgrid" / f"{cif_path.stem}").absolute(), array)


def process_multi_cif(cif_paths, target_dir: Path):
    for cif_path in tqdm(cif_paths):
        try:
            buildAtomGridi(cif_path, target_dir)
        except Exception as e:
            err(f"Error when processing {cif_path.stem}: {e}")


def buildAtomGrid(
    root_dir: str, cif_dir: str = "cif", target_dir: str = "spbnet", n_process: int = 1
):

    root_dir = Path(".") / root_dir
    cif_dir = root_dir / cif_dir
    target_dir = root_dir / target_dir

    cif_paths = list(cif_dir.iterdir())
    cif_paths = list(filter(lambda item: item.suffix == ".cif", cif_paths))
    # cif_paths = cif_paths[:100]

    title("BUILDING ATOM GRID")
    param(
        cif_dir=cif_dir.absolute(),
        target_dir=target_dir.absolute(),
        cif_num=len(cif_paths),
    )

    process_num = n_process

    cif_num_per_process = len(cif_paths) // process_num + 1
    processes = []
    for i in range(process_num):
        cif_paths_i = cif_paths[i * cif_num_per_process : (i + 1) * cif_num_per_process]
        process = Process(target=process_multi_cif, args=(cif_paths_i, target_dir))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    # for cif_path in tqdm(list(target_dir.iterdir())):
    #     buildAtomGridi(cif_path, target_dir)
    title("BUILD ATOM GRID END")

    return
