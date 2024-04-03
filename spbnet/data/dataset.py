import pickle
import pandas as pd
from pathlib import Path

import numpy as np

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        nbr_fea_len: int,
        task: str,
        draw_false_grid=False,
    ):
        super().__init__()
        filter_df = df.dropna(subset=[task])
        self.df = filter_df
        self.task = task
        self.data_dir = data_dir
        self.draw_false_grid = draw_false_grid
        self.nbr_fea_len = nbr_fea_len

        self.GRID = 30

        self.max_atom_num = 512

    def __len__(self):
        return len(self.df)

    # grid
    @staticmethod
    def handle_griddata_float(
        file_griddata, emin=-5000.0, emax=5000, bins=101, GRID=30, channel=20
    ):
        griddata = np.fromfile(file_griddata, dtype=np.float32)
        # griddata = griddata.astype(np.float16)  # [30 * 30 * 30 * 20]
        griddata = torch.from_numpy(griddata)
        griddata = griddata / 250 * 101 + 1

        griddata = griddata.reshape(GRID, GRID, GRID, channel)
        # lj = griddata[:, :, :, 12] - griddata[:, :, :, 19]
        lj = griddata[:, :, :, 12]
        correction = torch.cat(
            [griddata[:, :, :, :12], griddata[:, :, :, 13:19]], dim=-1
        )

        # griddata[griddata > 6e4] = 6e4
        # griddata[griddata < -6e4] = -6e4

        lj[lj <= emin] = emin
        lj[lj > emax] = emax

        correction[correction <= emin] = emin
        correction[correction > emax] = emax

        # x = np.linspace(emin, emax, bins)
        # lj = np.digitize(lj, x) + 1
        # correction = np.digitize(correction, x) + 1

        return lj, correction

    @staticmethod
    def handle_griddata(file_griddata: Path, GRID=30, channel=20):
        griddata = np.load(file_griddata.absolute())  # [30*30*30, 20] uint8
        griddata = griddata.astype(np.float32)

        griddata = griddata.reshape(GRID, GRID, GRID, channel)
        lj = griddata[:, :, :, 12]
        correction = np.concatenate(
            [griddata[:, :, :, :12], griddata[:, :, :, 13:19]], axis=-1
        )

        return torch.from_numpy(lj).float(), torch.from_numpy(correction).float()

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(np.abs(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_))

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cifid):
        file_grid = self.data_dir / "grid" / f"{cifid}.grid"
        file_griddata = self.data_dir / "griddata8" / f"{cifid}.npy"

        # get grid
        with file_grid.open("r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        lj, corr = self.handle_griddata(file_griddata)

        return cell, volume, lj, corr

    def get_grid_data(self, cifid, draw_false_grid=False):
        cell, volume, lj, corr = self.get_raw_grid_data(cifid)
        return {"cell": cell, "volume": volume, "lj": lj, "corr": corr}

    @staticmethod
    def get_gaussian_distance(distances, num_step, dmax, dmin=0, var=0.2):
        """
        Expands the distance by Gaussian basis
        (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        _filter = np.linspace(
            dmin, dmax, num_step
        )  # = np.arange(dmin, dmax + step, step) with step = 0.2

        return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var**2).float()

    # @lru_cache(maxsize=None)
    def get_graph(self, cifid: str):
        # moftransformer
        file_graph = self.data_dir / "graphdata" / f"{cifid}.graphdata"

        graphdata = pickle.load(file_graph.open("rb"))
        # graphdata = ["cifid", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
        atom_num = torch.LongTensor(graphdata[1].copy())
        nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
        nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

        nbr_fea = torch.FloatTensor(
            self.get_gaussian_distance(nbr_dist, num_step=self.nbr_fea_len, dmax=8)
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

    def get_coulomb(self, cifid: str):
        file_coulomb = self.data_dir / "coulomb" / f"{cifid}.coulomb"
        potential = torch.load(file_coulomb, map_location="cpu")
        return {"coulomb": potential}

    def __getitem__(self, index):
        ret = dict()

        item = self.df.iloc[index]
        cifid = item["cifid"]
        target = float(item[self.task])

        ret.update(
            {
                "cifid": cifid,
                "target": target,
            }
        )
        # ret.update(self.get_atom(cifid))
        ret.update(self.get_grid_data(cifid, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cifid))
        # ret.update(self.get_coulomb(cifid))

        # ret.update(self.get_tasks(index))

        return ret

    @staticmethod
    def collate(batch, img_size):
        """
        collate batch
        Args:
            batch (dict): [cifid, atom_num, nbr_idx, nbr_fea, uni_idx, uni_count,
                            grid_data, cell, (false_grid_data, false_cell), target]
            img_size (int): maximum length of img size

        Returns:
            dict_batch (dict): [cifid, atom_num, nbr_idx, nbr_fea, crystal_atom_idx,
                                uni_idx, uni_count, grid, false_grid_data, target]
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # graph
        # moftransformer
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

        # target
        batch_target = dict_batch["target"]
        batch_target = torch.tensor(batch_target, dtype=torch.float)
        dict_batch["target"] = batch_target

        # # coulomb
        # batch_coulomb = dict_batch["coulomb"]  # List[[H, W, D]]
        # batch_coulomb = torch.stack(batch_coulomb)  # [B, H, W, D]
        # # batch_coulomb = torch.clamp(batch_coulomb, min=None, max=300)
        # # batch_coulomb = normalize(batch_coulomb)
        # dict_batch["coulomb"] = batch_coulomb

        # grid
        batch_lj_data = dict_batch["lj"]
        batch_lj_data = torch.stack(batch_lj_data)
        batch_lj_data = batch_lj_data.view(batch_lj_data.shape[0], 30, 30, 30)
        dict_batch["lj"] = batch_lj_data

        batch_corr_data = dict_batch["corr"]
        batch_corr_data = torch.stack(batch_corr_data)
        batch_corr_data = batch_corr_data.view(batch_corr_data.shape[0], 30, 30, 30, 18)
        dict_batch["corr"] = batch_corr_data

        return dict_batch
