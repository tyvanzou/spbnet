from typing import Any, Optional
from functools import partial
from torch import nn
import pytorch_lightning as pl
import pandas as pd
import json
import yaml
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from multiprocessing import Process
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange
from torch import optim
import numpy as np
import click
import joblib

from ..datamodule.dataset import FeatDataset as Dataset
from ..modules.module import SpbNet
from ..modules.heads import RegressionHead, Pooler, ClassificationHead
from ..modules.optimize import set_scheduler
from ..modules import objectives
from ..utils.echo import title, start, end, param, err

cur_dir = Path(__file__).parent


class SpbNetTrainer(pl.LightningModule):
    def __init__(self, config: dict):
        super(SpbNetTrainer, self).__init__()
        self.save_hyperparameters()

        self.config = config

        self.model = SpbNet(config)
        # print(self.model)
        # self.model = AtomFormer(config)

        # pooler
        self.pooler = Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # void fraction prediction
        if 'vf' in self.config['feat']['save']:
            self.vfp_head = RegressionHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        # topo classify
        if "tc" in self.config["feat"]["save"]:
            assert config.get("topo_num") is not None
            self.tc_head = ClassificationHead(config["hid_dim"], config["topo_num"])
            self.tc_head.apply(objectives.init_weights)

        # atom grid classify
        self.agc_head = nn.Linear(config["hid_dim"], 1)
        self.agc_head.apply(objectives.init_weights)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # mae
        self.min_mae = 1e5

    def on_test_epoch_start(self) -> None:
        self.cifids = []

        self.cls_contents = []
        self.tc_contents = []
        self.vf_contents = []
        self.structure_contents = []
        self.potential_contents = []
        self.attns = []

        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        feat = self.model(batch)
        self.cifids += feat["cifid"]
        batch_size = len(feat["cifid"])

        # item
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        cls_feat_tensor = cls_feat.clone()
        cls_feat = cls_feat.detach().cpu().numpy()

        fconfig = self.config["feat"]

        if 'tc' in fconfig['save']:
            tc_pred = self.tc_head(cls_feat_tensor)  # [B, N_Topos]
            tc_pred = torch.argmax(tc_pred, dim=-1).reshape(-1).detach().cpu().numpy()
            self.tc_contents.append({
                "tc_pred": tc_pred
            })
        
        if 'vf' in fconfig['save']:
            vf_pred = self.vfp_head(cls_feat_tensor).reshape(-1).detach().cpu().numpy()  # [B]
            self.vf_contents.append({
                "vf_pred": vf_pred
            })

        if "cls" in fconfig["save"]:
            cls_content = {
                "cls_feat": cls_feat,
            }
            self.cls_contents.append(cls_content)

        if "structure" in fconfig["save"]:
            sfconfig = fconfig["structure"]
            structure_content = dict()

            structure_feat = (
                feat["structure_feat"].detach().cpu().numpy()
            )  # [B, max_graph_len, hid_dim]
            structure_mask = (
                feat["structure_mask"].detach().cpu().numpy()
            )  # [B, max_graph_len]
            atom_num = feat["atom_num"].detach().cpu().numpy()  # [B, max_graph_len]
            atom_attn = (
                feat["ca_attn"][:, 0].detach().cpu().numpy()
            )  # [B, max_graph_len]

            structure_feat = structure_feat.reshape(
                (-1, structure_feat.shape[-1])
            )  # [B * max_graph_len, hid_dim]

            if sfconfig["feat"]:
                structure_content["feat"] = structure_feat

            # sample
            if sfconfig["sample"]:
                atom_num = atom_num.reshape(-1)  # [B * max_graph_len]
                atom_attn = atom_attn.reshape(-1)  # [B * max_graph_len]
                structure_mask = structure_mask.reshape(-1)  # [B * max_graph_len]

                indices_of_ones = np.where(structure_mask == 0)[0]

                structure_indices = np.random.choice(
                    indices_of_ones,
                    size=sfconfig["sample_num_per_crystal"] * batch_size,
                    replace=False,
                )
                structure_feat = structure_feat[structure_indices]  # [20 * B, hid_dim]
                atom_num = atom_num[structure_indices]  # [20 * B, hid_dim]
                atom_attn = atom_attn[structure_indices]  # [20 * B, hid_dim]
                structure_content['atom_num'] = atom_num # [B, max_graph_len]
                structure_content["attn"] = atom_attn
            else:
                # atom_num = atom_num[indices_of_ones]
                # atom_attn = atom_attn[indices_of_ones]
                structure_content['atom_num'] = atom_num # [B, max_graph_len]
                structure_content['atom_mask'] = structure_mask # [B, max_graph_len]
                structure_content["attn"] = atom_attn

            self.structure_contents.append(structure_content)

        if "potential" in fconfig["save"]:
            pfconfig = fconfig["potential"]
            patch_size = self.config["patch_size"]["lj"]
            potential_feat = feat["potential_feat"]
            agc_pred = self.agc_head(potential_feat)

            lj = batch["lj"]  # [B, H, W, D]
            lj = lj.detach().cpu().numpy()
            lj = rearrange(
                agc_label,
                "b (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
            )
            lj = np.sum(
                agc_label, axis=-1
            )  # [B, GRID / PatchSize * GRID / PatchSize * GRID / PatchSize]

            agc_pred = (
                agc_pred.detach().cpu().numpy().reshape((batch_size, -1))
            )  # [B, 1004]
            potential_feat = potential_feat.detach().cpu().numpy()
            atomgrid = batch["atomgrid"].detach().cpu().numpy()
            agc_label = atomgrid
            agc_label = agc_label.transpose(-1, -3)  # [b x y z] -> [b h w d]
            agc_label = rearrange(
                agc_label,
                "b (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
            )
            agc_label = np.sum(
                agc_label, axis=-1
            )  # [B, GRID / PatchSize * GRID / PatchSize * GRID / PatchSize]
            potential_attn = (
                feat["ca_attn"][:, 0, 2:-2].detach().cpu().numpy()
            )  # [B, n_token]

            potential_feat = potential_feat.reshape((-1, potential_feat.shape[-1]))
            agc_pred = agc_pred.reshape(-1)
            agc_label = agc_label.reshape(-1)
            potential_attn = potential_attn.reshape(-1)

            if pfconfig["sample"]:
                potential_indices = np.random.randint(
                    0,
                    potential_feat.shape[0],
                    pfconfig["sample_num_per_crystal"] * batch_size,
                )
                potential_feat = potential_feat[potential_indices]
                agc_pred = agc_pred[potential_indices]
                agc_label = agc_label[potential_indices]
                potential_attn = potential_attn[potential_indices]
                lj = lj[potential_indices]

            potential_content = dict()
            if pfconfig["feat"]:
                potential_content["feat"] = potential_feat
            if pfconfig["attn"]:
                potential_content["attn"] = potential_attn
            if pfconfig["agc"]:
                potential_content["agc_label"] = agc_label
                potential_content["agc_pred"] = agc_pred
            if pfconfig["value"]:
                potential_content["lj"] = lj

            self.potential_contents.append(potential_content)

        if "attn" in fconfig["save"]:
            self_attn = feat["sa_attn"].detach().cpu().numpy()  # [B, 1004, 1004]
            cross_attn = (
                feat["ca_attn"].detach().cpu().numpy()
            )  # [B, 1004, max_graph_len]

            if fconfig["self_attn"]["sample"]:
                # attn
                attn_indices = np.random.randint(
                    0, batch_size, fconfig["self_attn"]["sample_num_per_batch"]
                )
                self_attn = self_attn[attn_indices, 2:-2, 2:-2]  # [S, 1000, 1000]
                cross_attn = cross_attn[attn_indices, 2:-2]  # [S, 1000, max_graph_len]

            attn_content = {"self_attn": self_attn, "cross_attn": cross_attn}

            self.attn_contents.append(attn_content)

    def on_test_epoch_end(self) -> None:
        save_dir = Path(self.config["save_dir"])
        save_dir.mkdir(exist_ok=True, parents=True)

        fconfig = self.config["feat"]

        def collate(contents):
            # contents: List[dict]
            assert len(contents) > 0

            ret = dict()
            keys = list(contents[0].keys())
            for key in keys:
                ret[key] = np.concatenate([content[key] for content in contents])
            return ret

        print(fconfig)

        # joblib.dump(self.cifids, str(save_dir / 'cifid.joblib'))
        cifid_df = pd.DataFrame([[cifid] for cifid in self.cifids], columns=['cifid'])
        cifid_df.to_csv(save_dir / 'cifid.csv', index=False)
        # with (save_dir / 'cifid.json').open('w') as f:
        #     json.dump(self.cifids, f)
        for feat_name in fconfig["save"]:
            if feat_name == "cls":
                joblib.dump(
                    collate(self.cls_contents), str(save_dir / f"cls.joblib")
                )  # [N * 20, hid_dim]
            if feat_name == "tc":
                joblib.dump(
                    collate(self.tc_contents), str(save_dir / f"tc.joblib")
                )  # [N * 20, hid_dim]
            if feat_name == "vf":
                joblib.dump(
                    collate(self.vf_contents), str(save_dir / f"vf.joblib")
                )  # [N * 20, hid_dim]
            if feat_name == "potential":
                joblib.dump(
                    collate(self.potential_contents),
                    str(save_dir / f"potential.joblib"),
                )  # [N * 20, hid_dim]
            if feat_name == "structure":
                joblib.dump(
                    collate(self.structure_contents),
                    str(save_dir / f"structure.joblib"),
                )  # [N * 20, hid_dim]
            if feat_name == "attn":
                joblib.dump(
                    collate(self.attn_contents), str(save_dir / f"attn.joblib")
                )  # [N * 20, hid_dim]

        return super().on_test_epoch_end()


def feat(config_path: Path):
    torch.set_float32_matmul_precision("medium")

    title("START TO OBTAIN FEATURES")

    model_config = yaml.full_load(
        (cur_dir / "../configs" / "config.model.yaml").open("r")
    )
    optimize_config = yaml.full_load(
        (cur_dir / "../configs" / "config.optimize.yaml").open("r")
    )
    default_train_config = yaml.full_load(
        (cur_dir / "../configs" / "config.feat.yaml").open("r")
    )

    with config_path.open("r") as f:
        user_config: dict = yaml.full_load(f)

    if user_config.get("root_dir") is None:
        err(f"Please specify root directory `root_dir`")
        return
    if user_config.get("ckpt") is None:
        err(f"Please specify ckpt `ckpt`")
        return
    if user_config.get("id_prop") is None:
        warn(f"Label data `id_prop` not specified, use default `benchmark.test`")
    if user_config.get("log_dir") is None:
        warn(f"Log directory not specified, use default `./lightning_logs/feat`")

    config = {**model_config, **optimize_config, **default_train_config, **user_config}

    # check
    device = config["device"]

    # handle
    root_dir = Path(config["root_dir"])
    id_prop = config["id_prop"]
    id_prop_path = root_dir / f"{id_prop}.csv"
    modal_dir = root_dir / config["modal_folder"]
    device = config["device"]
    log_dir = Path(config["log_dir"])

    # split train & val
    df = pd.read_csv(str(id_prop_path))

    dataset = Dataset(
        df=df,
        modal_dir=modal_dir,
        nbr_fea_len=config["nbr_fea_len"],
        useBasis=config["useBasis"],
        useCharge=config["useCharge"],
        img_size=config["img_size"]["lj"],
        feat_config=config["feat"],
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )

    # train
    logger = TensorBoardLogger(save_dir=(log_dir).absolute(), name="")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        min_epochs=0,
        devices=device,
        accelerator=config["accelerator"],
        strategy=config["strategy"],
        callbacks=[],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        precision=config["precision"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
    )

    ckpt_path = config["ckpt"]
    ckpt = torch.load(ckpt_path, weights_only=True)
    hparams = ckpt["hyper_parameters"]["config"]
    config["topo_num"] = hparams.get("topo_num")

    model = SpbNetTrainer(config)

    loadret = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("Load return", loadret)

    trainer.test(
        model=model,
        dataloaders=loader,
    )


@click.command()
@click.option(
    "--config-path", "-C", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def feat_cli(config_path: Path):
    feat(config_path)


if __name__ == "__main__":
    feat_cli()
