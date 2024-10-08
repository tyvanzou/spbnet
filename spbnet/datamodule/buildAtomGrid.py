from pathlib import Path
import math
import numpy as np
from ase.io import read
import argparse
from multiprocessing import Process
from tqdm import tqdm
import click

from .prepare_data import _make_supercell
from ..utils.echo import title, param, err


GRID = 30


def get_atoms(cif_path: Path):
    atoms = read(cif_path.absolute())
    atoms = _make_supercell(atoms, cutoff=30)
    return atoms


def cif2array(cif_path: Path):
    global GRID

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
    if (tgt_dir / "atomgrid" / f"{cif_path.stem}.npy").exists():
        return
    array = cif2array(cif_path)  # [GRID, GRID, GRID]
    np.save((tgt_dir / "atomgrid" / f"{cif_path.stem}").absolute(), array)


def process_multi_cif(cif_paths, moda_dir: Path):
    for cif_path in tqdm(cif_paths):
        try:
            buildAtomGridi(cif_path, moda_dir)
        except Exception as e:
            err(f"Error when processing {cif_path.stem}: {e}")


def buildAtomGrid(
    root_dir: Path,
    cif_folder: str = "cif",
    moda_folder: str = "spbnet",
    n_process: int = 1,
    grid: int = 30
):
    global GRID
    GRID = grid

    cif_dir = root_dir / cif_folder
    modal_dir = root_dir / modal_folder
    modal_dir.mkdir(exist_ok=True, parents=True)

    cif_paths = list(cif_dir.iterdir())
    cif_paths = list(filter(lambda item: item.suffix == ".cif", cif_paths))
    # cif_paths = cif_paths[:100]

    title("building atom grid")
    param(
        cif_dir=cif_dir.absolute(),
        modal_dir=modal_dir.absolute(),
        cif_num=len(cif_paths),
    )

    process_num = n_process

    cif_num_per_process = len(cif_paths) // process_num + 1
    processes = []
    for i in range(process_num):
        cif_paths_i = cif_paths[i * cif_num_per_process : (i + 1) * cif_num_per_process]
        process = Process(target=process_multi_cif, args=(cif_paths_i, moda_dir))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    # for cif_path in tqdm(list(moda_dir.iterdir())):
    #     buildAtomGridi(cif_path, moda_dir)
    title("building atom grid")

    return


@click.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--cif-folder", "-C", type=str, default="cif")
@click.option("--moda-folder", "-T", type=str, default="spbnet")
@click.option("--n-process", "-N", type=int, default=1)
@click.option("--grid", "-G", type=int, default=30)
def buildAtomGridCli(
    root_dir: Path, cif_folder: str, moda_folder: str, n_process: int, grid: int
):
    buildAtomGrid(root_dir, cif_folder, moda_folder, n_process, grid)


if __name__ == "__main__":
    buildAtomGridCli()
