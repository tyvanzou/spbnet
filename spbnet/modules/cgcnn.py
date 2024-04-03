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


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        max_graph_len=512,
        n_h=1,
        drop_ratio=0,
    ):
        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()

        self.max_graph_len = max_graph_len
        self.pad_embedding = nn.Embedding(1, h_fea_len)

        self.drop_ratio = drop_ratio
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )

        self.fc_head = nn.Linear(atom_fea_len, h_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        # crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        # crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        # crys_fea = self.conv_to_fc_softplus(crys_fea)

        return self.fc_head(atom_fea)

    def reconstruct_batch(self, atom_fea: torch.Tensor, crystal_atom_idx: torch.Tensor):
        """
        Params:
            atom_fea: [N0, h_dim]
            crystal_atom_idx: [N0]
        Return:
            atom_fea: [B, max_atom_len, h_dim]
            atom_fea_mask: [B, max_atom_len]
        """
        crystal_num = len(crystal_atom_idx)
        new_atom_fea = [[] for _ in range(crystal_num)]
        new_atom_fea_pad_mask = [None for _ in range(crystal_num)]
        for crystal_idx, atom_idxs in enumerate(crystal_atom_idx):
            for atom_idx in atom_idxs:
                new_atom_fea[crystal_idx].append(atom_fea[atom_idx])
        pad_embed = self.pad_embedding(
            torch.zeros([1], dtype=torch.long, device=atom_fea.device)
        ).view(
            -1
        )  # [h_fea_len]
        for crystal_idx, crystal_atom_fea in enumerate(new_atom_fea):
            if len(crystal_atom_fea) > self.max_graph_len:
                crystal_atom_fea = crystal_atom_fea[: self.max_graph_len]
                new_atom_fea[crystal_idx] = torch.stack(crystal_atom_fea).to(atom_fea)
                new_atom_fea_pad_mask[crystal_idx] = torch.zeros(
                    [self.max_graph_len]
                ).to(atom_fea)
            else:
                mask = torch.cat(
                    [
                        torch.zeros(len(crystal_atom_fea)),
                        torch.ones(self.max_graph_len - len(crystal_atom_fea)),
                    ]
                ).to(atom_fea)
                new_atom_fea_pad_mask[crystal_idx] = mask
                while len(crystal_atom_fea) < self.max_graph_len:
                    crystal_atom_fea.append(pad_embed.clone())
                new_atom_fea[crystal_idx] = torch.stack(crystal_atom_fea).to(atom_fea)
        new_atom_fea = torch.stack(new_atom_fea).to(atom_fea)
        new_atom_fea_pad_mask = torch.stack(new_atom_fea_pad_mask).to(atom_fea)
        return new_atom_fea, new_atom_fea_pad_mask


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

        new_atom_fea, new_atom_fea_pad_mask = self.reconstruct_batch(
            atom_fea, crystal_atom_idx
        )
        # [B, max_graph_len, hid_dim], [B, max_graph_len]
        return (
            new_atom_fea,
            new_atom_fea_pad_mask,
        )  # None will be replaced with MOC

    def reconstruct_batch(self, atom_fea, crystal_atom_idx):
        # return new_atom_fea, mask, mo_label
        batch_size = len(crystal_atom_idx)

        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, self.hid_dim], fill_value=0.0
        ).to(atom_fea)
        new_atom_fea_pad_mask = torch.full(
            size=[batch_size, self.max_graph_len], fill_value=0, dtype=torch.int
        ).to(atom_fea)
        for crystal_idx, atom_idxs in enumerate(crystal_atom_idx):
            if len(atom_idxs) < self.max_graph_len:
                new_atom_fea_pad_mask[crystal_idx] = torch.cat(
                    [
                        torch.zeros([len(atom_idxs)], dtype=torch.int),
                        torch.ones(
                            [self.max_graph_len - len(atom_idxs)], dtype=torch.int
                        ),
                    ]
                ).to(atom_fea)
            else:
                new_atom_fea_pad_mask[crystal_idx] = torch.zeros(
                    [self.max_graph_len], dtype=torch.int
                ).to(atom_fea)
            idx_ = atom_idxs[: self.max_graph_len]
            new_atom_fea[crystal_idx][: len(idx_)] = atom_fea[idx_]

        return new_atom_fea, new_atom_fea_pad_mask
