# MOFTransformer version 2.0.0
import random

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Args:
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns:

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """

        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # [N, M, atom_fea_len]

        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                # [N, atom_fea_len] -> [N, M, atom_fea_len] -> v_i
                atom_nbr_fea,  # [N, M, atom_fea_len] -> v_j
                nbr_fea,
            ],  # [N, M, nbr_fea_len] -> u(i,j)_k
            dim=2,
        )
        # [N, M, atom_fea_len*2+nrb_fea_len]

        total_gated_fea = self.fc_full(total_nbr_fea)  # [N, M, atom_fea_len*2]
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(
            N, M, self.atom_fea_len * 2
        )  # [N, M, atom_fea_len*2]
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # [N, M, atom_fea_len]
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # [N, atom_fea_len]
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)  # [N, atom_fea_len]
        return out


class GraphEmbeddings(nn.Module):
    """
    Generate Embedding layers made by only convolution layers of CGCNN (not pooling)
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(
        self, atom_fea_len, nbr_fea_len, max_graph_len, hid_dim, n_conv=3, vis=False
    ):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_graph_len = max_graph_len
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(119, atom_fea_len)  # 119 -> max(atomic number)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.fc = nn.Linear(atom_fea_len, hid_dim)

        self.vis = vis

    def forward(
        self,
        atom_num,
        nbr_idx,
        nbr_fea,
        crystal_atom_idx,
        uni_idx,
        uni_count,
        moc=None,
        charge=None,
    ):
        """
        Args:
            atom_num (tensor): [N', atom_fea_len]
            nbr_idx (tensor): [N', M]
            nbr_fea (tensor): [N', M, nbr_fea_len]
            crystal_atom_idx (list): [B]
            uni_idx (list) : [B]
            uni_count (list) : [B]
        Returns:
            new_atom_fea (tensor): [B, max_graph_len, hid_dim]
            mask (tensor): [B, max_graph_len]
        """
        assert self.nbr_fea_len == nbr_fea.shape[-1]

        atom_fea = self.embedding(atom_num)  # [N', atom_fea_len]
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)  # [N', atom_fea_len]
        atom_fea = self.fc(atom_fea)  # [N', hid_dim]

        new_atom_num, new_atom_fea, new_atom_fea_pad_mask, mo_labels, charges = (
            self.reconstruct_batch(atom_num, atom_fea, crystal_atom_idx, moc, charge)
        )
        # [B, max_graph_len, hid_dim], [B, max_graph_len]
        return (
            new_atom_num,
            new_atom_fea,
            new_atom_fea_pad_mask,
            mo_labels,
            charges,
        )  # None will be replaced with MOC

    def reconstruct_batch(self, atom_num, atom_fea, crystal_atom_idx, moc, charge):
        # return new_atom_fea, mask, mo_label
        batch_size = len(crystal_atom_idx)
        device = atom_fea.device

        new_atom_fea = torch.zeros(
            size=[batch_size, self.max_graph_len, self.hid_dim], device=device
        )
        new_atom_fea_pad_mask = torch.zeros(
            size=[batch_size, self.max_graph_len], device=device, dtype=torch.bool
        )
        new_atom_num = torch.zeros(
            size=[batch_size, self.max_graph_len], device=device, dtype=torch.int
        )
        if moc is not None:
            mo_labels = torch.zeros(
                size=[batch_size, self.max_graph_len], device=device, dtype=torch.int
            )
        else:
            mo_labels = None
        if charge is not None:
            charges = torch.zeros(
                size=[batch_size, self.max_graph_len], device=device
            ).to(device=device)
        else:
            charges = None
        for crystal_idx, atom_idxs in enumerate(crystal_atom_idx):
            if len(atom_idxs) < self.max_graph_len:
                new_atom_fea_pad_mask[crystal_idx] = torch.cat(
                    [
                        torch.zeros([len(atom_idxs)], dtype=torch.int),
                        torch.ones(
                            [self.max_graph_len - len(atom_idxs)], dtype=torch.int
                        ),
                    ]
                ).to(device=device, dtype=torch.bool)
                if moc is not None:
                    mo_labels[crystal_idx, moc[crystal_idx]] = 1
                    mo_labels[crystal_idx, len(atom_idxs) :] = -100
            else:
                new_atom_fea_pad_mask[crystal_idx] = torch.zeros(
                    [self.max_graph_len], dtype=int
                ).to(device=device, dtype=torch.bool)
                if moc is not None:
                    molabel = torch.LongTensor(moc[crystal_idx])  # List[int]
                    molabel = molabel[torch.where(molabel < self.max_graph_len)]
                    mo_labels[crystal_idx, molabel] = 1
            idx_ = atom_idxs[: self.max_graph_len]
            new_atom_fea[crystal_idx][: len(idx_)] = atom_fea[idx_]
            new_atom_num[crystal_idx][: len(idx_)] = atom_num[idx_]
            if charge is not None:
                charges[crystal_idx][: len(idx_)] = charge[idx_]

        return new_atom_num, new_atom_fea, new_atom_fea_pad_mask, mo_labels, charges
