from .prepare_data import make_prepared_data
from pathlib import Path
import math
import numpy as np
from ase.build import make_supercell
from ase.io import read
import shutil
import argparse
from spbnet.utils.echo import err, title, start, end


GRID = 30


def handle_griddata(file_griddata, emin=-5000.0, emax=5000, bins=101):
    griddata = np.fromfile(file_griddata.absolute(), dtype=np.float32)
    griddata = griddata.reshape(30 * 30 * 30, 20)

    griddata[griddata <= emin] = emin
    griddata[griddata > emax] = emax

    x = np.linspace(emin, emax, bins)
    griddata = np.digitize(griddata, x) + 1

    griddata = griddata.astype(np.uint8)

    return griddata


def get_atoms(cif_path: Path, tgt_dir: Path):
    try:

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

        cifpath = cif_path
        cifid = cif_path.stem
        atoms = read(cifpath.absolute())
        atoms = _make_supercell(atoms, cutoff=30)
        atoms.write(tgt_dir / f"energycell/{cifid}.cif")
        atoms = read(cifpath.absolute())
        atoms = _make_supercell(atoms, cutoff=8)
        atoms.write(tgt_dir / f"supercell/{cifid}.cif")
    except Exception as e:
        raise ValueError("Build Modal Data: get atoms error", e)
    return atoms


def buildGraphAndGrid(cif_path: Path, tgt_dir: Path):
    cifpath = cif_path
    cifid = cifpath.stem
    total_dir = tgt_dir / "total"
    total_dir.mkdir(exist_ok=True)
    cifid = cifpath.stem
    make_prepared_data(cifpath, total_dir)
    # move to target folder
    for suffix in ["graphdata", "grid", "griddata"]:
        (tgt_dir / suffix).mkdir(exist_ok=True)
        if not (total_dir / f"{cifid}.{suffix}").exists():
            continue
        shutil.copy2(
            (total_dir / f"{cifid}.{suffix}"), (tgt_dir / suffix / f"{cifid}.{suffix}")
        )


def buildGriddata8(cif_path: Path, tgt_dir: Path):
    cifid = cif_path.stem
    griddata8 = handle_griddata(tgt_dir / f"griddata/{cifid}.griddata")
    np.save((tgt_dir / f"griddata8/{cifid}").absolute(), griddata8)


def buildModalData(cif_path: Path, tgt_dir: Path):
    cifid = cif_path.stem

    def ifProcessed():
        suffix_map = {
            "graphdata": "graphdata",
            "grid": "grid",
            "griddata": "griddata",
            "griddata8": "npy",
            "supercell": "cif",
            "energycell": "cif",
        }

        for suffix in [
            "graphdata",
            "grid",
            "griddata",
            "griddata8",
            "supercell",
            "energycell",
        ]:
            if not (tgt_dir / suffix / f"{cifid}.{suffix_map[suffix]}").exists():
                print(f"{suffix} not prepared")
                return False
        if not (tgt_dir / "supercell" / f"{cifid}.cif").exists():
            print(f"super cell not prepared")
            return False
        return True

    title(f"START TO BUILD MODAL DATA - {cifid}")
    start("Star to check if processed")
    if ifProcessed():
        title("HAVE PROCESSED")
        return
    end("Check processed end")
    for suffix in [
        "graphdata",
        "grid",
        "griddata",
        "griddata8",
        "supercell",
        "energycell",
        "mol",
        "attn",
    ]:
        (tgt_dir / suffix).mkdir(exist_ok=True, parents=True)
    start("Start to build graph and grid")
    buildGraphAndGrid(cif_path, tgt_dir)
    start("End to build graph and grid")
    start("Start to build griddata8")
    buildGriddata8(cif_path, tgt_dir)
    start("End to build griddata8")
    start("Start to get atoms")
    get_atoms(cif_path, tgt_dir)
    start("End to get atoms")
    title("BUILD MODAL DATA END")
