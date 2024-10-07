import pickle
import pandas as pd
from typing import List

import numpy as np

import torch
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

import json


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        modal_dir: Path,
        nbr_fea_len: int,
        useBasis: bool = True,
        img_size: int = 30,
    ):
        """
        Dataset for MOF.
        Args:
            df (pd.DataFrame): cifid, mofid, target
            modal_dir (str): where dataset cif files and energy grid file; exist via model.utils.prepare_data.py
            split (str) : train, test, split
            nbr_fea_len (int) : nbr_fea_len for gaussian expansion
        """
        super().__init__()

        self.nbr_fea_len = nbr_fea_len
        self.useBasis = useBasis
        self.img_size = img_size

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def load_base_modals(self, cifid):
        ret = {"cifid": cifid}
        ret.update(self.load_griddata(cifid))
        ret.update(self.load_grid(cifid))
        ret.update(self.load_graph(cifid))

        return ret

    def load_griddata(self, cifid: str):
        file_griddata = self.modal_dir / "griddata8" / f"{cifid}.npy"

        griddata = np.load(file_griddata.absolute())  # [30*30*30, 20] uint8
        griddata = griddata.astype(np.float32)

        if self.useBasis:
            griddata = griddata.reshape(
                self.img_size, self.img_size, self.img_size, 20
            )  # 20 channel of basis functions, 13 for pauli, 7 for london, ignore coulomb
            lj = griddata[:, :, :, 12]
            correction = np.concatenate(
                [griddata[:, :, :, :12], griddata[:, :, :, 13:19]], axis=-1
            )

            return {
                "lj": torch.from_numpy(lj).float(),
                "corr": torch.from_numpy(correction).float(),
            }
        else:
            griddata = griddata.reshape(img_size, img_size, img_size)

            return {
                "lj": torch.from_numpy(griddata).float(),
                "corr": None,
            }

    def load_grid(self, cifid):
        def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
            a_ = np.cos(angle_a * np.pi / 180)
            b_ = np.cos(angle_b * np.pi / 180)
            c_ = np.cos(angle_c * np.pi / 180)

            v = (
                a
                * b
                * c
                * np.sqrt(np.abs(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_))
            )

            return v.item() / (60 * 60 * 60)  # normalized volume

        file_grid = self.modal_dir / "grid" / f"{cifid}.grid"

        # get grid
        with file_grid.open("r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        return {"cell": cell, "volume": volume}

    def load_graph(self, cifid: str):
        def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
            assert dmin < dmax
            _filter = np.linspace(
                dmin, dmax, num_step
            )  # = np.arange(dmin, dmax + step, step) with step = 0.2

            return np.exp(
                -((distances[..., np.newaxis] - _filter) ** 2) / var**2
            ).float()

        # moftransformer
        file_graph = self.modal_dir / "graphdata" / f"{cifid}.graphdata"

        graphdata = pickle.load(file_graph.open("rb"))
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
        )

        uni_idx = graphdata[4]
        uni_count = graphdata[5]

        return {
            "atom_num": atom_num,
            "nbr_idx": nbr_idx,
            "nbr_fea": nbr_fea,
            "uni_idx": uni_idx,
            "uni_count": uni_count,
        }

    def load_atomgrid(self, cifid: str):
        file_atomgrid = self.modal_dir / "atomgrid" / f"{cifid}.npy"
        arr = np.load(file_atomgrid)
        tensor = torch.from_numpy(arr).float()
        # tensor = torch.load(file_atomgrid)
        return {"atomgrid": tensor}

    def load_coulomb(self, cifid: str):
        file_coulomb = self.modal_dir / "coulomb" / f"{cifid}.coulomb"
        potential = torch.load(file_coulomb, map_location="cpu")
        return {"coulomb": potential}

    def load_charge(self, cifid: str):
        return {
            "charge": torch.from_numpy(
                np.load(self.modal_dir / "charge" / f"{cifid}.npy")
            ).float()
        }  # [N_atoms]

    def collate_fn(self, batch, modals: List[str]):
        """
        modals: available modals ['graph', 'grid', 'lj', 'corr', 'charge', 'coulomb', 'atomgrid']
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        if "graph" in modals:
            batch_atom_num = dict_batch["atom_num"]
            batch_nbr_idx = dict_batch["nbr_idx"]
            batch_nbr_fea = dict_batch["nbr_fea"]

            crystal_atom_idx = []
            base_idx = 0
            for i, nbr_idx in enumerate(batch_nbr_idx):
                n_i = nbr_idx.shape[0]
                crystal_atom_idx.append(torch.arange(n_i) + base_idx)
                nbr_idx += base_idx
                base_idx += n_i

            dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
            dict_batch["nbr_idx"] = torch.cat(batch_nbr_idx, dim=0)
            dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
            dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # griddata
        if "lj" in modals:
            batch_lj_data = dict_batch["lj"]
            batch_lj_data = torch.stack(batch_lj_data)
            batch_lj_data = batch_lj_data.view(
                batch_lj_data.shape[0], self.img_size, self.img_size, self.img_size
            )
            dict_batch["lj"] = batch_lj_data

        if "corr" in modals:
            batch_corr_data = dict_batch["corr"]
            batch_corr_data = torch.stack(batch_corr_data)
            batch_corr_data = batch_corr_data.view(
                batch_corr_data.shape[0],
                self.img_size,
                self.img_size,
                self.img_size,
                18,
            )
            dict_batch["corr"] = batch_corr_data

        # grid
        # default collate

        # atom grid
        if "atomgrid" in modals:
            batch_atomgrid = dict_batch["atomgrid"]
            batch_atomgrid = torch.stack(batch_atomgrid, dim=0)
            dict_batch["atomgrid"] = (
                batch_atomgrid  # [B, img_size, img_size, img_size] aka [B, 30, 30, 30]
            )

        # coulomb
        if "coulomb" in modals:
            batch_coulomb = dict_batch["coulomb"]  # List[[H, W, D]]
            batch_coulomb = torch.stack(batch_coulomb)  # [B, H, W, D]
            batch_coulomb.unsqueeze_(1)  # [B, 1, H, W, D]
            dict_batch["coulomb"] = batch_coulomb

        # charge
        if "charge" in modals:
            dict_batch["charge"] = torch.cat(
                dict_batch["charge"], dim=0
            )  # [N_atoms] same as atom_num

        return dict_batch


class FinetuneDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        modal_dir: Path,
        nbr_fea_len: int,
        task: str=None,
        task_type: str = "regression",
        useBasis: bool = True,
        useCharge: bool = False,
        img_size: int = 30,
        isPredict: bool = False,
    ):
        super().__init__(
            modal_dir=modal_dir,
            nbr_fea_len=nbr_fea_len,
            useBasis=useBasis,
            img_size=img_size,
        )

        if task is None and not isPredict:
            raise ValueError(f"FinetuneDataset: task is None while isPredict==False")

        if not isPredict:
            self.df = df.dropna(subset=[task])
        else:
            self.df = df
        self.modal_dir = modal_dir
        self.task = task
        self.task_type = task_type

        self.useCharge = useCharge
        self.useBasis = useBasis

        self.isPredict = isPredict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        cifid = item["cifid"]
        ret = self.load_base_modals(cifid)

        if not self.isPredict:
            ret.update({"target": item[self.task]})

        if self.useCharge:
            ret.update(self.load_charge(cifid))

        return ret

    def collate_fn(self, batch):
        modals = ["graph", "grid", "lj"]
        if self.useBasis:
            modals.append("corr")
        if self.useCharge:
            modals.append("charge")

        dict_batch = super().collate_fn(batch, modals)

        # target
        if not self.isPredict:
            batch_target = dict_batch["target"]
            if self.task_type == "regression":
                batch_target = torch.tensor(batch_target, dtype=torch.float)
            elif self.task_type == "classification":
                pass
            dict_batch["target"] = batch_target  # tensor or List[str]

        return dict_batch


class PretrainDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        modal_dir: Path,
        nbr_fea_len: int,
        useBasis: bool = True,
        useCharge: bool = False,
        useMoc: bool = False,
        img_size: int = 30,
    ):
        super().__init__(
            modal_dir=modal_dir,
            nbr_fea_len=nbr_fea_len,
            useBasis=useBasis,
            img_size=img_size,
        )

        self.df = df.dropna(subset["topo", "voidfraction"])
        self.useMoc = useMoc
        self.useBasis = useBasis
        self.useCharge = useCharge

        # Topo
        self.topo2tid = None
        topofile = modal_dir / "topo2tid.json"
        if topofile.exists():
            with topofile.open("r") as f:
                self.topo2tid = json.load(f)
        else:
            topo2tid = dict()
            for item in self.df.iloc:
                topo2tid[item["topo"]] = True
            self.topo2tid = dict()
            for tid, topo in enumerate(topo2tid.keys()):
                self.topo2tid[topo] = tid
            with topofile.open("w") as f:
                json.dump(self.topo2tid, f)

        if self.useMoc:
            with (modal_dir / "moc.json").open("r") as f:
                self.cifid2moc = json.load(
                    f
                )  # map[str, List[int] (index of metal atoms)]

    def get_topo_num(self):
        return len(list(self.topo2tid.keys()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        cifid = item["cifid"]
        topo = item["topo"]
        voidfraction = item["voidfraction"]

        ret = self.load_base_modals(cifid)
        ret.update({"topo": self.topo2tid[topo], "voidfraction": voidfraction})

        if self.useMoc:
            ret.update({"moc": self.cifid2moc[cifid]})

        return ret

    def collate_fn(self, batch):
        modals = ["graph", "grid", "lj", "atomgrid"]

        if self.useBasis:
            modals.append("corr")
        if self.useCharge:
            modals.append("charge")

        dict_batch = super().collate_fn(batch, modals)

        batch_topo = dict_batch["topo"]
        batch_topo = torch.tensor(batch_topo, dtype=torch.long)
        dict_batch["topo"] = batch_topo
        batch_voidfraction = dict_batch["voidfraction"]
        batch_voidfraction = torch.tensor(batch_voidfraction, dtype=torch.float)
        dict_batch["voidfraction"] = batch_voidfraction

        # moc
        # default collate fn

        return dict_batch


class FeatDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        modal_dir: Path,
        nbr_fea_len: int,
        feats: List[str],
        useBasis: bool = True,
        useCharge: bool = False,
        img_size: int = 30,
    ):
        super().__init__(
            df=df,
            modal_dir=modal_dir,
            nbr_fea_len=nbr_fea_len,
            useBasis=useBasis,
            img_size=img_size,
        )

        self.df = df  # since not target, here we do not dropna
        self.modal_dir = modal_dir
        self.feats = feats
        self.useBasis = useBasis
        self.useCharge = useCharge

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        cifid = item["cifid"]

        ret = self.load_base_modals(cifid)

        if "agc_pred" in self.feats or "agc_label" in self.feats:
            ret.update(self.load_atomgrid(cifid))

        return ret

    def collate_fn(self, batch):
        modals = ["graph", "grid", "lj"]

        if self.useBasis:
            modals.append("corr")
        if self.useCharge:
            modals.append("charge")

        if "agc_pred" in self.feats or "agc_label" in self.feats:
            modals.append("atomgrid")

        dict_batch = super().collate_fn(batch, modals)

        return dict_batch
