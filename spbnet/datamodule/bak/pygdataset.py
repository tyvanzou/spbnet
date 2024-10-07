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


class CifGraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        egFmt: str = "raw",
        target: Optional[str] = None,
        transform=None,
        pre_transform=None,
    ):
        self.egFmt = egFmt

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
        return ["filter_cifs.csv"] + [
            osp.join("cif", f"{cifid}.cif") for cifid in cifids
        ]

    @property
    def processed_file_names(self):
        raw_dir = Path(self.raw_dir)
        cifids = self.df["cifid"]
        return [osp.join("geodata", f"{cifid}.pt") for cifid in cifids] + [
            osp.join(f"griddata{self.egFmt}", f"{cifid}.npy") for cifid in cifids
        ]

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
        volume = self.calculate_volume(*lattice_constant)
        geodata["volume"] = torch.FloatTensor([volume])
        # print("DEBUG: ", geodata)
        griddata = torch.from_numpy(
            np.load(processed_dir / f"griddata{self.egFmt}" / f"{cifid}.npy")
        )

        # for finetune
        if self.isFinetune:
            target = self.df[self.target][idx]
            return {
                "geometric": geodata,
                "potential": griddata,
                "cifid": cifid,
                "target": torch.FloatTensor([target]),
            }
        else:
            return {"geometric": geodata, "potential": griddata, "cifid": cifid}

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(np.abs(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_))

        return v.item() / (60 * 60 * 60)  # normalized volume

    @staticmethod
    def add_virtual_nodes(
        data: torch_geometric.data.Data, VNODES_GRID: int = 10, VNODES_Z: int = 101
    ):
        lattice_constant = data["lattice_constant"].tolist()
        pos = data["pos"]
        X = data["x"]
        y = data.get("y")
        volume = data["volume"]

        lattice_obj = Lattice.from_parameters(*lattice_constant)
        grid_poses = get_grid_poses(lattice_obj, VNODES_GRID)
        pos_added = torch.concat([pos, grid_poses], dim=0)
        X_added = torch.concat(
            [
                X,
                torch.ones([VNODES_GRID * VNODES_GRID * VNODES_GRID], dtype=torch.long)
                * VNODES_Z,
            ],
            dim=0,
        )

        data_with_vnodes = Data(
            x=X_added,
            y=y.float() if y is not None else 0.0,
            pos=pos_added.float(),
            volume=volume,
            lattice_constant=torch.tensor(lattice_constant, dtype=torch.float),
        )

        return data_with_vnodes

    @staticmethod
    def collate(datalist, VNODES_GRID, VNODES_Z, isFinetune=False):
        geodatas = [data["geometric"] for data in datalist]
        potnetialdatas = [data["potential"] for data in datalist]
        cifids = [data["cifid"] for data in datalist]
        geodatas = [
            CifGraphDataset.add_virtual_nodes(geodata, VNODES_GRID, VNODES_Z)
            for geodata in geodatas
        ]
        # geo_collate_fn = Collater(dataset)
        batch_geodatas = Batch.from_data_list(geodatas)
        batch_potential_datas = torch.stack(potnetialdatas).float()

        if isFinetune:
            batch_target = torch.stack([data["target"] for data in datalist])
            return {
                "geometric": batch_geodatas,
                "potential": batch_potential_datas,
                "cifids": cifids,
                "target": batch_target,
            }
        else:
            return {
                "geometric": batch_geodatas,
                "potential": batch_potential_datas,
                "cifids": cifids,
            }

    def process(self):
        raw_dir = Path(self.raw_dir)
        # df = df[:100]
        # cifids = df["cifid"]
        # data_list = []

        for item in tqdm(self.df.iloc, total=len(self.df)):
            cifid = item["cifid"]
            # target = item["N2-77-100000"]
            atoms = read(str(raw_dir / "cif" / f"{cifid}.cif"))
            atoms = _make_supercell(
                atoms, cutoff=30.0
            )  # to be consistent with energy supercell

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


def get_grid_poses(lat, GRID: int = 30):
    grid_poses = []
    for i in range(GRID):
        for j in range(GRID):
            for k in range(GRID):
                grid_pos = lat.get_cartesian_coords([i / GRID, j / GRID, k / GRID])
                grid_poses.append(torch.from_numpy(grid_pos).float())
    grid_poses = torch.stack(grid_poses)  # [N, 3]
    return grid_poses


@click.command()
@click.option("--root-dir", "-R", type=str)
def init(root_dir):
    df = pd.read_csv(osp.join(root_dir, "filter_cifs.csv"))
    dataset = CifGraphDataset(root_dir, df, egFmt="i8")
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(CifGraphDataset.collate, VNODES_GRID=30, VNODES_Z=101),
    )
    for batch in loader:
        geodata = batch["geometric"]
        griddata = batch["potential"]
        print(geodata, griddata.shape)
        break
    # print(dataset)
    # print(len(dataset))
    # print(dataset[0], dataset[1])


if __name__ == "__main__":
    init()
