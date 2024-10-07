from functools import partial
from openbabel import openbabel as ob
import json
from pathlib import Path
import click
import subprocess
from pymatgen.core.structure import Structure
from multiprocessing import Pool
from tqdm import tqdm


def cif2mol(cifid: str, root_dir: Path):
    cif_path = root_dir / "cif" / f"{cifid}.cif"
    mol_path = root_dir / "mol2" / f"{cifid}.mol2"
    subprocess.run(
        [
            "obabel",
            "-iCIF",
            str(cif_path),
            "-omol2",
            "-O",
            str(mol_path),
            "--partialcharge",
            "eqeq",
            "-h",
        ]
    )


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


@click.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, type=Path)
)
@click.option("--cif-folder", "-C", type=str, default="cif")
@click.option("--n-process", "-N", type=int, default=8)
def main(root_dir, cif_folder, n_process):
    cif_dir = root_dir / cif_folder
    cifids = [file.stem for file in cif_dir.iterdir()]
    # cifids = cifids[:20]

    with Pool(n_process) as pool:
        result = list(
            tqdm(pool.imap_unordered(partial(cif2mol, root_dir=root_dir), cifids))
        )

    with Pool(n_process) as pool:
        result = list(
            tqdm(pool.imap_unordered(partial(mol2json, root_dir=root_dir), cifids))
        )


if __name__ == "__main__":
    main()
