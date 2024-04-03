from typing import Any, Optional
from functools import partial

# from pytorch_lightning.utilities.types import STEP_OUTPUT
# from model.crossformer import CrossFormer
from spbnet.modules.module import CrossFormer
from spbnet.modules.heads import RegressionHead, Pooler, ClassificationHead
from spbnet.modules.optimize import set_scheduler
from spbnet.modules import objectives
from torch import nn
from spbnet.data.dataset_feat import Dataset
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
from spbnet.utils.echo import title, start, end, param, err

cur_dir = Path(__file__).parent


def r2_score(y_true, y_pred):
    # 计算总平均值
    mean_true = torch.mean(y_true)

    # 计算总平方和
    total_ss = torch.sum((y_true - mean_true) ** 2)

    # 计算残差平方和
    resid_ss = torch.sum((y_true - y_pred) ** 2)

    # 计算R2分数
    r2 = 1 - (resid_ss / total_ss)

    return r2


class CrossFormerTrainer(pl.LightningModule):
    def __init__(self, config: dict):
        super(CrossFormerTrainer, self).__init__()
        self.save_hyperparameters()

        self.config = config

        self.model_config = config
        self.optimizer_config = config

        model_config = self.model_config
        self.model = CrossFormer(model_config)
        # print(self.model)
        # self.model = AtomFormer(config)

        self.mean = config["mean"]
        self.std = config["std"]

        # pooler
        self.pooler = Pooler(model_config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # void fraction prediction
        self.vfp_head = RegressionHead(model_config["hid_dim"])
        self.vfp_head.apply(objectives.init_weights)

        # topo classify
        self.tc_head = ClassificationHead(model_config["hid_dim"], config["topo_num"])
        self.tc_head.apply(objectives.init_weights)

        # atom grid classify
        self.agc_head = nn.Linear(model_config["hid_dim"], 1)
        self.agc_head.apply(objectives.init_weights)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # mae
        self.min_mae = 1e5

    def on_test_epoch_start(self) -> None:
        self.cifids = []

        self.feats = []
        self.structure_feats = []
        self.potential_feats = []
        self.agc_preds = []
        self.agc_labels = []
        self.self_attns = []

        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        feat = self.model(batch)
        self.cifids += feat["cifid"]
        batch_size = len(feat["cifid"])

        # item
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat).detach().cpu().numpy()
        structure_feat = (
            feat["structure_feat"].detach().cpu().numpy()
        )  # [B, max_graph_len, hid_dim]
        structure_mask = (
            feat["structure_mask"].detach().cpu().numpy()
        )  # [B, max_graph_len]
        potential_feat = feat["potential_feat"]  # [B, 1004, hid_dim]
        agc_pred = (
            self.agc_head(potential_feat)
            .detach()
            .cpu()
            .numpy()
            .reshape((batch_size, -1))
        )  # [B, 1004]
        potential_feat = potential_feat.detach().cpu().numpy()
        atomgrid: np.array = batch["atomgrid"].detach().cpu().numpy()
        agc_label = atomgrid
        patch_size = self.model_config["patch_size"]["lj"]
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
        self_attn = feat["sa_attn"].detach().cpu().numpy()  # [B, 1004, 1004]
        cross_attn = feat["ca_attn"].detach().cpu().numpy()  # [B, 1004, max_graph_len]

        # sample
        # structure
        structure_feat = structure_feat.reshape(
            (-1, structure_feat.shape[-1])
        )  # [B * max_graph_len, hid_dim]
        structure_mask = structure_mask.reshape(-1)  # [B * max_graph_len]
        indices_of_ones = np.where(structure_mask == 0)[0]
        structure_indices = np.random.choice(
            indices_of_ones, size=20 * batch_size, replace=False
        )
        structure_feat = structure_feat[structure_indices]  # [20 * B, hid_dim]

        # potential
        potential_feat = potential_feat.reshape((-1, potential_feat.shape[-1]))
        agc_pred = agc_pred.reshape(-1)
        agc_label = agc_label.reshape(-1)
        potential_indices = np.random.randint(
            0, potential_feat.shape[0], 20 * batch_size
        )
        potential_feat = potential_feat[potential_indices]
        agc_pred = agc_pred[potential_indices]
        agc_label = agc_label[potential_indices]

        # attn
        attn_indices = np.random.randint(0, batch_size, 5)
        self_attn = self_attn[attn_indices, 2:-2, 2:-2]  # [5, 1000, 1000]

        for feat_name in self.config["feats"]:
            if feat_name == "potential":
                self.potential_feats.append(potential_feat)  # [20 * B, hid_dim]
            if feat_name == "structure":
                self.structure_feats.append(structure_feat)  # [20 * B, hid_dim]
            if feat_name == "agc_pred":
                self.agc_preds.append(agc_pred)  # [20 * B]
            if feat_name == "agc_label":
                self.agc_labels.append(agc_label)  # [20 * B]
            if feat_name == "feat":
                self.feats.append(feat)  # [B]
            if feat_name == "self_attn":
                self.self_attns.append(self_attn)  # [5, 1000, 1000]

    def on_test_epoch_end(self) -> None:
        save_dir = Path(self.config["save_dir"], parents=True)
        save_dir.mkdir(exist_ok=True)

        for feat_name in self.config["feats"]:
            if feat_name == "potential":
                potential_feats = np.concatenate(self.potential_feats)
                np.save(
                    save_dir / f"potential_feats.npy", potential_feats
                )  # [N * 20, hid_dim]
            if feat_name == "structure":
                structure_feats = np.concatenate(self.structure_feats)
                np.save(
                    save_dir / f"structure_feats.npy", structure_feats
                )  # [N * 20, hid_dim]
            if feat_name == "agc_pred":
                agc_preds = np.concatenate(self.agc_preds)
                np.save(save_dir / f"agc_preds.npy", agc_preds)  # [N * 20]
            if feat_name == "agc_label":
                agc_labels = np.concatenate(self.agc_labels)
                np.save(save_dir / f"agc_labels.npy", agc_labels)  # [N * 20]
            if feat_name == "feat":
                feats = np.concatenate(self.feats)
                np.save(save_dir / f"feats.npy", feats)  # [N, hid_dim]
            if feat_name == "self_attn":
                self_attns = np.concatenate(self.self_attns)
                np.save(
                    save_dir / f"self_attns.npy", self_attns
                )  # [N * 5 / B, 1000, 1000]
        return super().on_test_epoch_end()


def feat(config_path: str):
    title("START TO OBTAIN FEATURES")

    base_config_path = (cur_dir / "config.yaml").absolute()
    with open(base_config_path, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_path, "r") as f:
        user_config = yaml.load(f, Loader=yaml.FullLoader)

    if user_config.get("data_dir") is None:
        err(f"Please specify modal directory `data_dir`!")
        return
    if user_config.get("id_prop") is None:
        err(f"Please specify label data `id_prop`")
        return
    if user_config.get("task") is None:
        err(f"Please specify task")
        return
    if user_config.get("log_dir") is None:
        err(f"Please specify log directory")
        return
    if user_config.get("save_dir") is None:
        err(f"Please specify save directory")
        return

    base_config.update(user_config)
    config = base_config

    # check
    device = config["device"]
    task = config["task"]

    # handle
    id_prop_path = Path(config["id_prop"])
    ckpt_path = Path(config["ckpt"])  # TODO: CKPT
    data_dir = Path(config["data_dir"])
    log_dir = Path(config["log_dir"])

    # split train & val
    id_prop_dir = id_prop_path.parent
    test_df = pd.read_csv(id_prop_path.absolute(), dtype={"cifid": str})
    test_dataset = Dataset(test_df, data_dir, config["nbr_fea_len"], task)

    # id_prop
    id_prop_df = test_df.copy()
    filter_id_prop_df = id_prop_df.dropna(subset=[task])

    # mean & std
    mean = None
    std = None
    mean = filter_id_prop_df[task].mean()
    std = filter_id_prop_df[task].std()
    config["mean"] = float(mean)
    config["std"] = float(std)

    # train
    checkpoint_callback = ModelCheckpoint(monitor="val_mae", mode="min", save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=(log_dir / task).absolute(), name="")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        min_epochs=0,
        devices=device,
        accelerator=config["accelerator"],
        strategy=config["strategy"],
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        precision=config["precision"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
    )
    model = CrossFormerTrainer(config)
    if not config["ckpt"] == "scratch":
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"], strict=True)

    trainer.test(
        model=model,
        dataloaders=DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
        ),
        ckpt_path=None,
    )


def main():
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    tasks = config["tasks"]

    test(0, config_path)


if __name__ == "__main__":
    main()
