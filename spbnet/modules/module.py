# MOFTransformer version 2.1.0
import torch
import torch.nn as nn
import math
from torch.nn import (
    TransformerDecoder,
    # TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from . import objectives, heads
from .cgcnn import GraphEmbeddings
from .vision_transformer_3d import VisionTransformer3D
from .transformer import TransformerDecoderLayer


class SpbNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        self.config["potential"] = self.config["lj"] or self.config["coulomb"]

        self.max_grid_len = config["max_grid_len"]
        # self.vis = config["visualize"]

        # graph embedding
        self.graph_embeddings = GraphEmbeddings(
            atom_fea_len=config["atom_fea_len"],
            nbr_fea_len=config["nbr_fea_len"],
            max_graph_len=config["max_graph_len"],
            hid_dim=config["hid_dim"],
            vis=config["visualize"],
            n_conv=config["nlayers"]["conv"],
        )
        self.graph_embeddings.apply(objectives.init_weights)

        # energy embedding
        if config["lj"]:
            self.lj_vit3d = VisionTransformer3D(
                img_size=config["img_size"]["lj"],
                patch_size=config["patch_size"]["lj"],
                in_chans=config["in_chans"],
                embed_dim=config["hid_dim"],
                depth=1,
                num_heads=config["nhead"],
            )

        if config["coulomb"]:
            self.coulomb_vit3d = VisionTransformer3D(
                img_size=config["img_size"]["coulomb"],
                patch_size=config["patch_size"]["coulomb"],
                in_chans=config["in_chans"],
                embed_dim=config["hid_dim"],
                depth=1,
                num_heads=config["nhead"],
            )

        if config["structure"]:
            structure_encoder_layer = TransformerEncoderLayer(
                config["hid_dim"],
                config["nhead"],
                config["hid_dim"],
                config["dropout"],
                batch_first=True,
            )
            self.structure_encoder = TransformerEncoder(
                structure_encoder_layer, config["nlayers"]["structure"]
            )

        # potential
        if self.config["potential"]:
            potential_decoder_layer = TransformerDecoderLayer(
                config["hid_dim"],
                config["nhead"],
                config["hid_dim"],
                config["dropout"],
                batch_first=True,
            )
            self.potential_decoder = TransformerDecoder(
                potential_decoder_layer, config["nlayers"]["potential"]
            )
        if not self.config["structure"]:
            self.fix_structure_embedding = nn.Parameter(
                torch.randn(1, self.config["max_graph_len"], self.config["hid_dim"])
            )

        # potential useBasis
        if self.config["useBasis"]:
            self.potential_mapper = nn.Linear(18, 1, bias=False)
            self.potential_mapper.apply(objectives.init_weights)

        # token type embeddings
        # LJ & coulomb
        self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
        self.token_type_embeddings.apply(objectives.init_weights)

        # head
        self.pooler = heads.Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # class token
        self.cls_embeddings = nn.Linear(1, config["hid_dim"])
        self.cls_embeddings.apply(objectives.init_weights)

        # volume token
        self.volume_embeddings = nn.Linear(1, config["hid_dim"])
        self.volume_embeddings.apply(objectives.init_weights)

        # sep token
        self.sep_embeddings = nn.Linear(1, config["hid_dim"])
        self.sep_embeddings.apply(objectives.init_weights)

        # charge
        if self.config["useCharge"]:
            self.charge_embeddings = nn.Linear(1, config["hid_dim"])
            self.charge_embeddings.apply(objectives.init_weights)

        # # regression
        # self.regression_head = heads.RegressionHead(config["hid_dim"])

    def forward(self, batch, mask_grid=False):
        cifid = batch["cifid"]

        if self.config["structure"]:
            atom_num = batch["atom_num"]  # [N']
            nbr_idx = batch["nbr_idx"]  # [N', M]
            nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
            crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
            uni_idx = batch["uni_idx"]  # list [B]
            uni_count = batch["uni_count"]  # list [B]

        if self.config["lj"]:
            # grid = batch["grid"]  # [B, C, H, W, D, 20]
            lj = batch["lj"]  # [B, H, W, D]
            corr = batch["corr"]  # [B, H, W, D, 18]

        if self.config["coulomb"]:
            # coulomb = None
            coulomb = batch["coulomb"]  # [B, H, W, D]
            coulomb.unsqueeze_(1)  # [B, C, H, W, D]

        volume = batch["volume"]  # list [B]

        # grid useBasis
        if self.config["useBasis"]:
            corr = self.potential_mapper(corr).squeeze(-1)  # [B, H, W, D]
            grid = (lj + corr).unsqueeze(1)  # [B, 1, H, W, D]
        else:
            grid = lj
            grid = grid.unsqueeze(1)
        # grid = (bias + grid_delta).unsqueeze(1)

        # graph embeds
        (
            atom_num,  # [B, max_graph_len]
            graph_embed,  # [B, max_graph_len, hid_dim],
            graph_mask,  # [B, max_graph_len],
            mo_labels,  # [B, max_graph_len] default moc is None
            charges,  # [B, max_graph_len] default charge is None
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
            moc=batch.get("moc"),  # default moc is None
            charge=batch.get("charge"),  # default charge is None
        )

        if self.config["useCharge"]:
            charges.unsqueeze_(-1)
            charges_embed = self.charge_embeddings(charges)
            graph_embed += charges_embed

        if self.config["lj"]:
            # lj embed
            (
                lj_embed,  # [B, max_grid_len+1, hid_dim]
                lj_mask,  # [B, max_grid_len+1]
                _,  # [B, grid+1, C] if mask_image == True
            ) = self.lj_vit3d.visual_embed(
                grid,
                max_image_len=self.max_grid_len,
                mask_it=mask_grid,
            )

        if self.config["coulomb"]:
            # coulomb embed
            (
                coulomb_embed,  # [B, max_grid_len+1, hid_dim]
                coulomb_mask,  # [B, max_grid_len+1]
                _,  # [B, grid+1, C] if mask_image == True
            ) = self.coulomb_vit3d.visual_embed(
                coulomb,
                max_image_len=self.max_grid_len,
                mask_it=mask_grid,
            )

        # fusion
        batch_size = len(cifid)
        if self.config["lj"]:
            device = lj.device
        else:
            device = graph_embed.device

        # volume embed
        volume = torch.FloatTensor(volume).to(device)  # [B]
        volume_embed = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1).to(device)

        # cls_embed
        cls_token = torch.zeros(batch_size).to(device)  # [B]
        cls_embed = self.cls_embeddings(cls_token[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(batch_size, 1).to(device)  # [B, 1]

        # sep_embed
        sep_token = torch.zeros(batch_size).to(device)  # [B]
        sep_embed = self.sep_embeddings(sep_token[:, None, None])  # [B, 1, hid_dim]
        sep_mask = torch.ones(batch_size, 1).to(device)  # [B, 1]

        # add token_type_embeddings
        if self.config["lj"]:
            lj_embed = lj_embed + self.token_type_embeddings(
                torch.zeros_like(lj_mask, device=device).long()
            )
        if self.config["coulomb"]:
            coulomb_embed = coulomb_embed + self.token_type_embeddings(
                torch.ones_like(coulomb_mask, device=device).long()
            )

        # ablation
        if self.config["structure"] and not self.config["potential"]:
            graph_embed = torch.cat([cls_embed, graph_embed], dim=1)
            graph_mask = torch.cat([cls_mask, graph_mask], dim=1)
            feat = self.structure_encoder(graph_embed, src_key_padding_mask=graph_mask)
            cls_feat = feat[:, 0:1, :]
            return {
                "cifid": cifid,
                "cls_feat": cls_feat,
            }

        if self.config["potential"] and not self.config["structure"]:
            if self.config["lj"] and self.config["coulomb"]:
                potential_embed = torch.cat(
                    [
                        cls_embed,
                        lj_embed,
                        sep_embed,
                        coulomb_embed,
                        sep_embed,
                        volume_embed,
                    ],
                    dim=1,
                )
                potential_mask = torch.cat(
                    [cls_mask, lj_mask, sep_mask, coulomb_mask, sep_mask, volume_mask],
                    dim=1,
                )
            elif self.config["lj"]:
                potential_embed = torch.cat(
                    [cls_embed, lj_embed, sep_embed, volume_embed],
                    dim=1,
                )
                potential_mask = torch.cat(
                    [cls_mask, lj_mask, sep_mask, volume_mask], dim=1
                )  # [B, max_grid_len+2]
            else:
                potential_embed = torch.cat(
                    [cls_embed, coulomb_embed, sep_embed, volume_embed], dim=1
                )  # [B, max_grid_len+2, hid_dim]
                potential_mask = torch.cat(
                    [cls_mask, coulomb_mask, sep_mask, volume_mask], dim=1
                )  # [B, max_grid_len+2]

            structure_mem = self.fix_structure_embedding.repeat(
                [batch_size, 1, 1]
            )  # [B, max_graph_len, hid_dim]
            structure_mask = torch.zeros(*structure_mem.shape[:-1]).to(device)

            feat = self.potential_decoder(
                tgt=potential_embed,
                tgt_key_padding_mask=potential_mask,
                memory=structure_mem,
                memory_key_padding_mask=structure_mask,
                tgt_mask=None,
            )

            cls_feat = feat[:, 0:1, :]

            return {"cifid": cifid, "cls_feat": cls_feat}

        if self.config["lj"] and self.config["coulomb"]:
            # potential_embed = torch.cat(
            #     [
            #         cls_embed,
            #         lj_embed,
            #         sep_embed,
            #         coulomb_embed,
            #         sep_embed,
            #         volume_embed,
            #     ],
            #     dim=1,
            # )  # [B, max_grid_len+2, hid_dim]
            # potential_mask = torch.cat(
            #     [cls_mask, lj_mask, sep_mask, coulomb_mask, sep_mask, volume_mask],
            #     dim=1,
            # )  # [B, max_grid_len+2]
            # if lj_embed.shape != coulomb_embed.shape:
            #     raise ValueError(
            #         f"lj img {self.config['img_size']['lj'], self.config['patch_size']['lj']} while coulomb img {self.config['img_size']['coulomb'], self.config['patch_size']['coulomb']}"
            #     )
            # potential_embed = torch.cat(
            #     [cls_embed, lj_embed, sep_embed, coulomb_embed, sep_embed, volume_embed], dim=1
            # )
            # potential_mask = torch.cat(
            #     [cls_mask, lj_mask, sep_mask, coulomb_mask, sep_mask, volume_mask], dim=1
            # )
            # potential_end_idx = int(cls_embed.shape[1] + lj_embed.shape[1])
            potential_embed = lj_embed + coulomb_embed
            potential_mask = lj_mask.bool() | coulomb_mask.bool()
            potential_embed = torch.cat(
                [cls_embed, potential_embed, sep_embed, volume_embed], dim=1
            )
            potential_mask = torch.cat(
                [cls_mask, potential_mask, sep_mask, volume_mask], dim=1
            )
        elif self.config["lj"] and not self.config["coulomb"]:
            potential_embed = torch.cat(
                [cls_embed, lj_embed, sep_embed, volume_embed], dim=1
            )  # [B, max_grid_len+2, hid_dim]
            potential_mask = torch.cat(
                [cls_mask, lj_mask, sep_mask, volume_mask], dim=1
            )  # [B, max_grid_len+2]

        structure_mem, structure_mask = None, None
        structure_mem = self.structure_encoder(
            graph_embed, src_key_padding_mask=graph_mask
        )
        structure_mask = graph_mask

        sa_attn, ca_attn = None, None
        if not self.config["visualize"]:
            feat = self.potential_decoder(
                tgt=potential_embed,
                tgt_key_padding_mask=potential_mask,
                memory=structure_mem,
                memory_key_padding_mask=structure_mask,
                tgt_mask=None,
            )
        else:
            feat = potential_embed
            for idx, layer in enumerate(self.potential_decoder.layers):
                if idx == len(self.potential_decoder.layers) - 1:
                    x = feat
                    tgt_key_padding_mask = potential_mask
                    tgt_mask = None
                    x_add, sa_attn = layer._sa_block_attn(
                        x, tgt_mask, tgt_key_padding_mask
                    )
                    x = layer.norm1(x + x_add)
                    x_add, ca_attn = layer._mha_block_attn(
                        x,
                        structure_mem,
                        None,
                        structure_mask,
                    )
                    x = layer.norm2(x + x_add)
                    x = layer.norm3(x + layer._ff_block(x))
                    feat = x
                else:
                    feat = layer(
                        tgt=feat,
                        tgt_key_padding_mask=potential_mask,
                        memory=structure_mem,
                        memory_key_padding_mask=structure_mask,
                        tgt_mask=None,
                    )

        cls_feat = feat[:, 0:1, :]

        potential_feat = feat[
            :, 2:-2, :
        ]  # cls_embed, lj_cls_embed, sep_embed, volume_embed

        # return cls_feat, smiles_feat, smiles_token_mask
        return {
            "cifid": cifid,  # List
            "atom_num": atom_num,  # [B, max_graph_len]
            "cls_feat": cls_feat,  # [B, max_graph_len, hid_dim]
            "sa_attn": sa_attn,  # [B, grid_all_len, grid_len] NOTE grid_all_len include cls and volume token etc..
            "ca_attn": ca_attn,  # [B, grid_all_len, graph_len] NOTE grid_all_len include cls and volume token etc..
            "potential_feat": potential_feat,
            "structure_feat": structure_mem,
            "structure_mask": graph_mask,
            "mo_labels": mo_labels,
        }
