from functools import partial
from pathlib import Path
from typing import Optional
import pandas as pd
import torch_geometric
import torch_geometric.data
from tqdm import tqdm
import click
import os
import os.path as osp
import math
import yaml

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data import Batch
from torch_geometric.data import InMemoryDataset, Dataset

from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import make_supercell
from ase.io import read


class CifGraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        target: Optional[str] = None,
        transform=None,
        pre_transform=None,
    ):
        # filter
        if target is not None:
            self.isFinetune = True
            self.df = df.dropna(subset=[target])
            self.target = target
        else:
            self.isFinetune = False
            self.df = df

        Path(root, "geodata").mkdir(exist_ok=True)

        super(CifGraphDataset, self).__init__(root, transform, pre_transform)
        # self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root)

    @property
    def raw_file_names(self):
        cifids = self.df["cifid"]
        return ["benchmark.csv"] + [osp.join("cif", f"{cifid}.cif") for cifid in cifids]

    @property
    def processed_file_names(self):
        raw_dir = Path(self.raw_dir)
        cifids = self.df["cifid"]
        return [osp.join("geodata", f"{cifid}.pt") for cifid in cifids]

    def len(self):
        return len(self.df)

    def get(self, idx):
        raw_dir = Path(self.raw_dir)
        processed_dir = Path(self.processed_dir)
        cifids = self.df["cifid"]
        cifid = cifids[idx]
        geodata = torch.load(
            processed_dir / "geodata" / f"{cifid}.pt", weights_only=False
        )
        lattice_constant = geodata["lattice_constant"].tolist()

        # for finetune
        if self.isFinetune:
            target = self.df[self.target][idx]
            return {
                "geometric": geodata,
                "cifid": cifid,
                "target": torch.FloatTensor([target]),
            }
        else:
            return {"geometric": geodata, "cifid": cifid}

    @staticmethod
    def collate(datalist, isFinetune=False):
        geodatas = [data["geometric"] for data in datalist]
        cifids = [data["cifid"] for data in datalist]
        # geo_collate_fn = Collater(dataset)
        batch_geodatas = Batch.from_data_list(geodatas)

        if isFinetune:
            batch_target = torch.stack([data["target"] for data in datalist])
            return {
                "geometric": batch_geodatas,
                "cifids": cifids,
                "target": batch_target,
            }
        else:
            return {
                "geometric": batch_geodatas,
                "cifids": cifids,
            }

    def process(self):
        raw_dir = Path(self.raw_dir)
        (raw_dir / "geodata").mkdir(exist_ok=True)
        # df = df[:100]
        # cifids = df["cifid"]
        # data_list = []

        for item in tqdm(self.df.iloc, total=len(self.df)):
            cifid = item["cifid"]
            # target = item["N2-77-100000"]
            atoms = read(str(raw_dir / "cif" / f"{cifid}.cif"))

            structure = AseAtomsAdaptor.get_structure(atoms)
            lat = structure.lattice
            lattice_constant = [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma]
            atomic_numbers = [site.specie.Z for site in structure]
            positions = np.stack([site.coords for site in structure])  # [N, 3] numpy

            data = Data(
                x=torch.tensor(atomic_numbers, dtype=torch.long),
                pos=torch.from_numpy(positions).to(torch.float),
                lattice_constant=torch.tensor(lattice_constant, dtype=torch.float),
                # y=torch.tensor([target], dtype=torch.float),
            )

            torch.save(data, raw_dir / "geodata" / f"{cifid}.pt")

            # data_list.append(data)

        # self.save(data_list, self.processed_paths[0])


def init():
    config = yaml.full_load(open("./config.yaml", "r"))
    id_prop = config["id_prop"]
    root_dir = Path(config["root_dir"])
    df = pd.read_csv(osp.join(root_dir, f"{id_prop}.csv"))
    target = config["target"]

    dataset = CifGraphDataset(root_dir, df, target=target)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(CifGraphDataset.collate, isFinetune=target is not None),
    )
    for batch in loader:
        geodata = batch["geometric"]
        print(geodata, batch["target"])
        break
    # print(dataset)
    # print(len(dataset))
    # print(dataset[0], dataset[1])


if __name__ == "__main__":
    init()
