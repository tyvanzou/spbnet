import pickle
import pandas as pd
from pathlib import Path
import json
import numpy as np
import torch

from dataset import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        nbr_fea_len: int,
        task: str,
        task_type: str = "regression",
        useCharge: bool = False,
        useTopoHelp: bool = False,
        draw_false_grid=False,
    ):
        super().__init__()
        filter_df = df.dropna(subset=[task])
        self.df = filter_df
        self.task = task
        self.task_type = task_type
        self.data_dir = data_dir
        self.draw_false_grid = draw_false_grid
        self.nbr_fea_len = nbr_fea_len

        self.useCharge = useCharge
        self.useTopoHelp = useTopoHelp

        if self.useTopoHelp:
            self.cifid2topo = json.load((self.data_dir / "cifid2topo.json").open("r"))

    def __len__(self):
        return len(self.df)

    def get_topo(self, cifid: str):
        return {"topo": torch.LongTensor([self.cifid2topo[cifid]])}

    def __getitem__(self, index):
        ret = dict()

        item = self.df.iloc[index]
        cifid = item["cifid"]

        if self.task_type == "regression":
            target = float(item[self.task])  # float
        elif self.task_type == "classification":
            target = item[self.task]  # str
        else:
            raise ValueError(
                f"Task type only support regression, classification, not {self.task_type}"
            )

        ret.update(
            {
                "cifid": cifid,
                "target": target,
            }
        )
        # ret.update(self.get_atom(cifid))
        ret.update(self.get_grid_data(cifid, draw_false_grid=self.draw_false_grid))
        ret.update(self.get_graph(cifid))
        if self.useCharge:
            ret.update(self.get_charge(cifid))
        if self.useTopoHelp:
            ret.update(self.get_topo(cifid))
        # ret.update(self.get_coulomb(cifid))

        # ret.update(self.get_tasks(index))

        return ret

    @staticmethod
    def collate(
        batch,
        img_size,
        task_type: str = "regression",
        useCharge=False,
        useTopoHelp=False,
    ):
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
        if useCharge:
            dict_batch["charge"] = torch.cat(
                dict_batch["charge"], dim=0
            )  # [N_atoms] same as atom_num
        if useTopoHelp:
            dict_batch["topo"] = torch.cat(dict_batch["topo"], dim=0)  # [B]
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        # target
        batch_target = dict_batch["target"]
        if task_type == "regression":
            batch_target = torch.tensor(batch_target, dtype=torch.float)
        elif task_type == "classification":
            pass
        dict_batch["target"] = batch_target  # tensor or List[str]

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
        # dict_batch['corr'] = None

        return dict_batch
