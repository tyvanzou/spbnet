import yaml
from pathlib import Path

from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell
from ase.io import read

import pandas as pd
import joblib
from tqdm import tqdm
import numpy as np


def main(root_dir, df):
    config = yaml.full_load(open("./config.yaml", "r"))

    root_dir = Path(config["root_dir"])
    id_prop = config["id_prop"]

    unimol_dir = root_dir / "unimol"
    unimol_dir.mkdir(exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        df = pd.read_csv(root_dir / f"{id_prop}.{split}.csv")
        custom_data = {"atoms": [], "coordinates": []}
        for item in tqdm(df.iloc, total=len(df)):
            cifid = item["cifid"]
            # target = item["N2-77-100000"]
            atoms = read(str(root_dir / "cif" / f"{cifid}.cif"))

            structure = AseAtomsAdaptor.get_structure(atoms)
            lat = structure.lattice
            lattice_constant = [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma]
            atomic_symbols = [site.specie.symbol for site in structure]
            positions = np.stack([site.coords for site in structure])  # [N, 3] numpy

            custom_data["atoms"].append(atomic_symbols)
            custom_data["coordinates"].append(positions)

        joblib.dump(custom_data, str(unimol_dir / f"{split}.joblib"))


if __name__ == "__main__":
    main()
