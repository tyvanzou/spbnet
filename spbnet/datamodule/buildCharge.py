from functools import partial
from openbabel import openbabel as ob
import json
from pathlib import Path
import click
import subprocess
from pymatgen.core.structure import Structure
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pickle
import torch

from ase.io import read
from .prepare_data import _make_supercell

from ..utils.echo import *


def cif2graphcif(cifid: str, root_dir: Path):
    cif_dir = root_dir / "cif"
    graphcif_dir = root_dir / "graphcif"
    if (graphcif_dir / f"{cifid}.cif").exists():
        warn(f"{str(graphcif_dir / f'{cifid}.cif')} already exists, return")
        return

    graphcif_dir.mkdir(exist_ok=True)

    atoms = read(str(cif_dir / f"{cifid}.cif"))
    atoms = _make_supercell(atoms, cutoff=8)
    atoms.write(filename=str(graphcif_dir / f"{cifid}.cif"))


def cif2mol(cifid: str, root_dir: Path):
    cif_path = root_dir / "graphcif" / f"{cifid}.cif"
    mol_dir = root_dir / "mol2"
    mol_dir.mkdir(exist_ok=True)
    mol_path = mol_dir / f"{cifid}.mol2"
    if mol_path.exists():
        with mol_path.open("r") as f:
            content: str = f.read()
        if content.strip() != "":
            warn(f"Non-empty { str(mol_path)} already exists, return")
            return

    charge_methods = ["eqeq", "qeq", "qtpie", "gasteiger", "mmff94", "eem"]
    for charge_method in charge_methods:
        if not mol_path.exists():
            subprocess.run(
                [
                    "obabel",
                    "-iCIF",
                    str(cif_path),
                    "-omol2",
                    "-O",
                    str(mol_path),
                    "--partialcharge",
                    charge_method,
                    "-h",
                ]
            )
        else:
            with mol_path.open("r") as f:
                content: str = f.read()
            if content.strip() == "":
                mol_path.unlink()
                subprocess.run(
                    [
                        "obabel",
                        "-iCIF",
                        str(cif_path),
                        "-omol2",
                        "-O",
                        str(mol_path),
                        "--partialcharge",
                        charge_method,
                        "-h",
                    ]
                )
            else:
                break


def mol2json(cifid: str, root_dir: Path):
    mol2_file = root_dir / "mol2" / f"{cifid}.mol2"
    obconversion = ob.OBConversion()
    obconversion.SetInAndOutFormats("mol2", "smi")  # 设置输入和输出格式
    mol = ob.OBMol()
    obconversion.ReadFile(mol, str(mol2_file))

    atom_list = []

    for i in range(mol.NumAtoms()):
        atom = mol.GetAtom(i + 1)
        # 获取原子的坐标和电荷信息
        coords = [atom.x(), atom.y(), atom.z()]
        charge = atom.GetPartialCharge()
        # 构建原子信息字典
        atom_info = {
            "atom": atom.GetAtomicNum(),  # 原子序数
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "charge": charge,
        }
        atom_list.append(atom_info)

    structure = Structure.from_file(str(root_dir / "cif" / f"{cifid}.cif"))

    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

    lattice_params = [a, b, c, alpha, beta, gamma]

    result = {"atom_list": atom_list, "lattice": lattice_params}

    with open(root_dir / "json" / f"{cifid}.json", "w") as f:
        json.dump(result, f, indent=4)


def mol2charge(cifid: str, root_dir: Path):
    charge_dir = root_dir / "charge"
    charge_dir.mkdir(exist_ok=True)
    charge_file = charge_dir / f"{cifid}.npy"
    if charge_file.exists():
        warn(f"{str(charge_file)} aleady exists, return")
        return

    mol2_file = root_dir / "mol2" / f"{cifid}.mol2"
    obconversion = ob.OBConversion()
    obconversion.SetInAndOutFormats("mol2", "smi")  # 设置输入和输出格式
    mol = ob.OBMol()
    obconversion.ReadFile(mol, str(mol2_file))

    atom_list = []

    charges = []
    for i in range(mol.NumAtoms()):
        atom = mol.GetAtom(i + 1)
        # 获取原子的坐标和电荷信息
        coords = [atom.x(), atom.y(), atom.z()]
        charge = atom.GetPartialCharge()
        charges.append(float(charge))

    charges = np.array(charges)
    np.save(charge_dir / f"{cifid}.npy", charges)


def check(cifid: str, root_dir: Path):
    with (root_dir / "graphdata" / f"{cifid}.graphdata").open("rb") as f:
        graphdata = pickle.load(f)
    atom_num = torch.LongTensor(graphdata[1].copy())
    charge = np.load(root_dir / f"charge/{cifid}.npy")
    if atom_num.shape[0] != charge.shape[0]:
        err(f"Err: {cifid}, charge shape and atom_num is not equal")
        return False
    else:
        return True


@click.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--n-process", "-N", type=int, default=8)
def main(root_dir, n_process):
    title("build charge, depend on openbabel")

    cif_dir = root_dir / "cif"

    cifids = [file.stem for file in cif_dir.iterdir() if file.suffix == ".cif"]
    # cifids = cifids[:20]

    param(
        root_dir=str(root_dir),
        n_process=n_process,
        len_cif_id=len(cifids),
        example=cifids[:4],
    )

    start(f"[1/4] build graph cif, target dir is {str(root_dir / 'graphcif')}")

    with Pool(n_process) as pool:
        result = list(
            tqdm(
                pool.imap_unordered(partial(cif2graphcif, root_dir=root_dir), cifids),
                total=len(cifids),
            )
        )

    start(
        f"[2/4] assign partial charge via openbabel, target dir is {str(root_dir / 'mol2')}"
    )

    with Pool(n_process) as pool:
        result = list(
            tqdm(
                pool.imap_unordered(partial(cif2mol, root_dir=root_dir), cifids),
                total=len(cifids),
            )
        )

    start(f"[3/4] format data, target dir is {str(root_dir / 'charge')}")

    with Pool(n_process) as pool:
        result = list(
            tqdm(
                pool.imap_unordered(partial(mol2charge, root_dir=root_dir), cifids),
                total=len(cifids),
            )
        )

    start(f"[4/4] check charge with graphdata")

    pbar = tqdm(cifids)
    err_count = 0
    err_cifids = []
    for cifid in tqdm(cifids):
        pbar.set_description(f"[{err_count}/{len(cifids)}]")
        result = check(cifid, root_dir)
        if not result:
            err_count += 1
            err_cifids.append(cifid)
    err(f"Charged Error: {err_count} cifs fail to generate charge, include {err_cifids}")

    end("process end")


if __name__ == "__main__":
    main()
