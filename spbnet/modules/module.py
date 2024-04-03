# MOFTransformer version 2.1.0
import torch
import torch.nn as nn

from spbnet.modules import objectives, heads
from spbnet.modules.cgcnn import GraphEmbeddings
from spbnet.modules.vision_transformer_3d import VisionTransformer3D
from spbnet.modules.transformer import TransformerDecoderLayer
from torch.nn import (
    TransformerDecoder,
    # TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

import math


class CrossFormer(nn.Module):
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
            # n_conv=config["nlayers"][0],
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
                # mlp_ratio=config["mlp_ratio"],
                # drop_rate=config["drop_rate"],
                # mpp_ratio=config["mpp_ratio"],
            )

        if config["coulomb"]:
            self.coulomb_vit3d = VisionTransformer3D(
                img_size=config["img_size"]["coulomb"],
                patch_size=config["patch_size"]["coulomb"],
                in_chans=config["in_chans"],
                embed_dim=config["hid_dim"],
                depth=1,
                num_heads=config["nhead"],
                # mlp_ratio=config["mlp_ratio"],
                # drop_rate=config["drop_rate"],
                # mpp_ratio=config["mpp_ratio"],
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
        if self.config["potential"] and not self.config["structure"]:
            potential_encoder_layer = TransformerEncoderLayer(
                config["hid_dim"],
                config["nhead"],
                config["hid_dim"],
                config["dropout"],
                batch_first=True,
            )
            self.potential_encoder = TransformerEncoder(
                potential_encoder_layer, config["nlayers"]["potential"]
            )
        elif self.config['potential']:
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

        # potential correction
        if self.config["correction"]:
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

        # # regression
        # self.regression_head = heads.RegressionHead(config["hid_dim"])

    def forward(
        self,
        batch,
        mask_grid=False,
    ):
        cifid = batch["cifid"]

        # atom_num_masked = batch["atom_num_masked"]  # [N']
        # atom_num_mask = batch["atom_num_mask"]  # [N']

        atom_num = batch["atom_num"]  # [N']
        nbr_idx = batch["nbr_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        uni_idx = batch["uni_idx"]  # list [B]
        uni_count = batch["uni_count"]  # list [B]

        # grid = batch["grid"]  # [B, C, H, W, D, 20]
        lj = batch["lj"]  # [B, H, W, D]
        corr = batch["corr"]  # [B, H, W, D, 18]
        volume = batch["volume"]  # list [B]

        coulomb = None
        # coulomb = batch.get("coulomb")  # [B, H, W, D]
        # coulomb.unsqueeze_(1) # [B, C, H, W, D]

        # grid correction
        if self.config["correction"]:
            corr = self.potential_mapper(corr).squeeze(-1)  # [B, H, W, D]
            grid = (lj + corr).unsqueeze(1)  # [B, 1, H, W, D]
        else:
            grid = lj
            grid = grid.unsqueeze(1)
        # grid = (bias + grid_delta).unsqueeze(1)

        # graph embeds
        (
            graph_embed,  # [B, max_graph_len, hid_dim],
            graph_mask,  # [B, max_graph_len],
        ) = self.graph_embeddings(
            atom_num=atom_num,
            nbr_idx=nbr_idx,
            nbr_fea=nbr_fea,
            crystal_atom_idx=crystal_atom_idx,
            uni_idx=uni_idx,
            uni_count=uni_count,
        )

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
        batch_size = lj_embed.shape[0]
        device = lj_embed.device

        # volume embed
        volume = torch.FloatTensor(volume).to(lj_embed)  # [B]
        volume_embed = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
        volume_mask = torch.ones(volume.shape[0], 1).to(lj_mask)

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
                )  # [B, max_grid_len+2]
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
            feat = self.potential_encoder(
                potential_embed, src_key_padding_mask=potential_mask
            )
            cls_feat = feat[:, 0:1, :]
            return {
                "cls_feat": cls_feat,
            }

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
            if lj_embed.shape != coulomb_embed.shape:
                raise ValueError(
                    f"lj img {self.config['img_size']['lj'], self.config['patch_size']['lj']} while coulomb img {self.config['img_size']['coulomb'], self.config['patch_size']['coulomb']}"
                )
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
            "cifid": cifid,
            "cls_feat": cls_feat,
            "sa_attn": sa_attn,
            "ca_attn": ca_attn,
            "potential_feat": potential_feat,
            "structure_feat": structure_mem,
            "structure_mask": graph_mask
        }

    def mask_tokens(self, src, mask_prob, src_padding_mask):
        mask = torch.zeros_like(src).bernoulli_(mask_prob).bool()
        if src_padding_mask is not None:
            mask = mask & ~src_padding_mask
        src = src.masked_fill(mask, 5)
        # mask = mask | src_padding_mask
        return mask, src

    def generate_square_subsequent_mask(self, sz, DEVICE="cuda"):
        diag = torch.triu(torch.ones((sz, sz), device=DEVICE), diagonal=1)
        diag[0:1, :] = torch.zeros([1, sz], device=DEVICE)
        diag[0, 0] = 1
        mask = diag == 1
        return mask

    def get_token_mask(self, src):
        padding_mask = src == self.PAD_IDX
        token_mask, src = self.mask_tokens(src, 0.01, padding_mask)
        return token_mask, padding_mask, src
