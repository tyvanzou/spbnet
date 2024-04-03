from spbnet.modules.module import CrossFormer
import yaml
from pathlib import Path
from ase.io import read
import math
import numpy as np
from ase.build import make_supercell
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import pickle
from ase.io import write
from pathlib import Path
import os
from ase.visualize.plot import plot_atoms


def get_grid_data(data_dir, cifid):
    def handle_griddata(file_griddata: Path, GRID=30, channel=20):
        griddata = np.load(file_griddata.absolute())  # [30*30*30, 20] uint8
        griddata = griddata.astype(np.float32)

        griddata = griddata.reshape(GRID, GRID, GRID, channel)
        # lj = griddata[:, :, :, 12] - griddata[:, :, :, 19]
        lj = griddata[:, :, :, 12]
        # ls = np.where(lj > 100, 0, lj)
        correction = np.concatenate(
            [griddata[:, :, :, :12], griddata[:, :, :, 13:19]], axis=-1
        )

        return torch.from_numpy(lj).float(), torch.from_numpy(correction).float()

    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_)

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(cifid):
        file_grid = data_dir / "grid" / f"{cifid}.grid"
        file_griddata = data_dir / "griddata8" / f"{cifid}.npy"

        # get grid
        with file_grid.open("r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        lj, corr = handle_griddata(file_griddata)

        return cell, volume, lj, corr

    def get_grid_data(cifid):
        cell, volume, lj, corr = get_raw_grid_data(cifid)
        return {"cell": cell, "volume": volume, "lj": lj, "corr": corr}

    return get_grid_data(cifid)


def get_graph(data_dir, cifid: str, nbr_fea_len=64):
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

    # moftransformer
    file_graph = data_dir / "graphdata" / f"{cifid}.graphdata"

    graphdata = pickle.load(file_graph.open("rb"))
    # graphdata = ["cifid", "atom_num", "nbr_idx", "nbr_dist", "uni_idx", "uni_count"]
    atom_num = torch.LongTensor(graphdata[1].copy())
    nbr_idx = torch.LongTensor(graphdata[2].copy()).view(len(atom_num), -1)
    nbr_dist = torch.FloatTensor(graphdata[3].copy()).view(len(atom_num), -1)

    nbr_fea = torch.FloatTensor(
        get_gaussian_distance(nbr_dist, num_step=nbr_fea_len, dmax=8)
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


def collate(batch):
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


def get_atoms(cif_file: Path, cutoff=8):
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

    atoms = read(str(cif_file.absolute()))
    atoms = _make_supercell(atoms, cutoff=cutoff)
    return atoms
