from typing import Any, Optional
from functools import partial
import click

from torch import optim, nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import yaml
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from einops import rearrange
from sklearn.model_selection import train_test_split
from pathlib import Path

from torchmetrics import Accuracy

from .datamodule.dataset import PretrainDataset as Dataset
from .modules.module import SpbNet
from .modules.heads import RegressionHead, ClassificationHead, Pooler
from .modules.optimize import set_scheduler
from .modules import objectives
from .utils.echo import err, title, start, end, param


cur_dir = Path(__file__).parent


class SpbNetTrainer(pl.LightningModule):
    def __init__(self, config: dict):
        super(SpbNetTrainer, self).__init__()
        self.save_hyperparameters()

        self.config = config

        self.model = SpbNet(config)

        # pooler
        self.pooler = Pooler(config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        # void fraction prediction
        self.vfp_head = RegressionHead(config["hid_dim"])
        self.vfp_head.apply(objectives.init_weights)

        # topo classify
        self.tc_head = ClassificationHead(config["hid_dim"], config["topo_num"])
        self.tc_head.apply(objectives.init_weights)

        # atom grid classify
        self.agc_head = nn.Linear(config["hid_dim"], 1)
        self.agc_head.apply(objectives.init_weights)

        if self.config["useMoc"]:
            self.moc_head = nn.Linear(config["hid_dim"], 1)
            self.moc_head.apply(objectives.init_weights)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.acc = Accuracy()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # lr log
        lrs = self.lr_schedulers().get_last_lr()
        lr_avg = sum(lrs)
        lr_avg /= len(lrs)
        self.log("lr", lr_avg)

        # [batch_size, max_token_len]
        vf = batch["voidfraction"]  # [B] float
        topo = batch["topo"]  # [B] int
        atomgrid = batch["atomgrid"]  # [B, GRID, GRID, GRID] aka [B, 30, 30, 30]

        feat = self.model(batch)

        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)

        # vfp
        vfp = self.vfp_head(cls_feat)  # [batch_size, 1]
        vfp = vfp.reshape(vfp.shape[0])  # [batch_size]
        vf = vf.reshape(vf.shape[0])  # [batch_size]
        vfp_mse = self.mse_loss(vfp, vf)
        vfp_mae = self.mae_loss(vfp, vf)
        self.log("train_vfp_mse", vfp_mse, batch_size=vf.shape[0], sync_dist=True)
        self.log("train_vfp_mae", vfp_mae, batch_size=vf.shape[0], sync_dist=True)

        # tc
        topo_pred = self.tc_head(cls_feat)  # [batch_size, topo_num]
        topo = topo.reshape(vf.shape[0])  # [batch_size]
        tc_loss = self.cross_entropy(topo_pred, topo)
        tc_acc = self.acc(topo, topo_pred)
        self.log("train_tc_loss", tc_loss, batch_size=topo.shape[0], sync_dist=True)
        self.log("train_tc_acc", tc_acc, batch_size=topo.shape[0], sync_dist=True)

        # atom grid
        potential_feat = feat[
            "potential_feat"
        ]  # [B, GRID / PathSize, GRID / PathSize, GRID / PathSize, hid_dim] aka [B, 10, 10, 10, 768]
        agc_pred = self.agc_head(potential_feat)
        agc_label: torch.Tensor = atomgrid
        patch_size = self.config["patch_size"]["lj"]
        # NOTE: Correct
        agc_label = agc_label.transpose(-1, -3)  # [b x y z] -> [b h w d]
        agc_label = rearrange(
            agc_label,
            "b (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3)",
            p1=patch_size,
            p2=patch_size,
            p3=patch_size,
        )
        agc_label = agc_label.sum(
            -1, keepdim=True
        )  # [B, GRID / PatchSize, GRID / PatchSize, GRID / PatchSize, 1]
        agc_mse = self.mse_loss(agc_label, agc_pred)
        agc_mae = self.mae_loss(agc_label, agc_pred)
        self.log(
            "train_agc_mse", agc_mse, batch_size=agc_label.shape[0], sync_dist=True
        )
        self.log(
            "train_agc_mae", agc_mae, batch_size=agc_label.shape[0], sync_dist=True
        )

        if self.config["useMoc"]:
            mo_labels = feat["mo_labels"].reshape(-1)  # [B, max_graph_len]
            moc_pred = self.moc_head(feat["structure_feat"]).reshape(
                -1
            )  # [B, max_graph_len]
            mask = mo_labels != -100
            mo_labels = mo_labels[mask]
            moc_pred = moc_pred[mask]
            moc_loss = F.binary_cross_entropy_with_logits(
                input=moc_pred, target=mo_labels
            )
            threshold = 0.5
            moc_pred = torch.where(moc_pred > 0.5, 1, 0)
            score = torch.where(moc_pred == mo_labels, 1.0, 0.0)
            acc = torch.mean(score)
            self.log(
                "train_moc_loss",
                moc_loss,
                batch_size=agc_label.shape[0],
                sync_dist=True,
            )
            self.log(
                "train_moc_acc", acc, batch_size=agc_label.shape[0], sync_dist=True
            )

        loss = (
            self.config["vfp"] * vfp_mse
            + self.config["tc"] * tc_loss
            + self.config["agc"] * agc_mse
        )
        if self.config["useMoc"]:
            loss += self.config["moc"] * moc_loss
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        # [batch_size, max_token_len]
        cifid = batch["cifid"]
        vf = batch["voidfraction"]  # [B] float
        topo = batch["topo"]  # [B] int
        atomgrid = batch["atomgrid"]  # [B, GRID, GRID, GRID] aka [B, 30, 30, 30]

        feat = self.model(batch)

        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)

        # vfp
        vfp = self.vfp_head(cls_feat)  # [batch_size, 1]
        vfp = vfp.reshape(vfp.shape[0])  # [batch_size]
        vf = vf.reshape(vf.shape[0])  # [batch_size]
        vfp_mse = self.mse_loss(vfp, vf)
        vfp_mae = self.mae_loss(vfp, vf)
        self.log("val_vfp_mse", vfp_mse, batch_size=vf.shape[0], sync_dist=True)
        self.log("val_vfp_mae", vfp_mae, batch_size=vf.shape[0], sync_dist=True)

        # tc
        topo_pred = self.tc_head(cls_feat)  # [batch_size, topo_num]
        topo = topo.reshape(vf.shape[0])  # [batch_size]
        tc_loss = self.cross_entropy(topo_pred, topo)
        # tc_loss = self.focal_loss(topo_pred, topo)
        tc_acc = self.cal_acc(topo, topo_pred)
        self.log("val_tc_loss", tc_loss, batch_size=topo.shape[0], sync_dist=True)
        self.log("val_tc_acc", tc_acc, batch_size=topo.shape[0], sync_dist=True)

        # atom grid classify
        potential_feat = feat[
            "potential_feat"
        ]  # [B, GRID / PathSize, GRID / PathSize, GRID / PathSize, hid_dim] aka [B, 10, 10, 10, 768]
        agc_pred = self.agc_head(potential_feat)
        agc_label: torch.Tensor = atomgrid
        patch_size = self.config["patch_size"]["lj"]
        agc_label = agc_label.transpose(-1, -3)  # [b x y z] -> [b h w d]
        agc_label = rearrange(
            agc_label,
            "b (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3)",
            p1=patch_size,
            p2=patch_size,
            p3=patch_size,
        )
        agc_label = agc_label.sum(
            -1, keepdim=True
        )  # [B, GRID / PatchSize, GRID / PatchSize, GRID / PatchSize, 1]
        agc_mse = self.mse_loss(agc_label, agc_pred)
        agc_mae = self.mae_loss(agc_label, agc_pred)
        self.log("val_agc_mse", agc_mse, batch_size=agc_label.shape[0], sync_dist=True)
        self.log("val_agc_mae", agc_mae, batch_size=agc_label.shape[0], sync_dist=True)

        if self.config["useMoc"]:
            mo_labels = feat["mo_labels"].reshape(-1)  # [B, max_graph_len]
            moc_pred = self.moc_head(feat["structure_feat"]).reshape(
                -1
            )  # [B, max_graph_len]
            mask = mo_labels != -100
            mo_labels = mo_labels[mask]
            moc_pred = moc_pred[mask]
            moc_loss = F.binary_cross_entropy_with_logits(
                input=moc_pred, target=mo_labels
            )
            threshold = 0.5
            moc_pred = torch.where(moc_pred > 0.5, 1, 0)
            score = torch.where(moc_pred == mo_labels, 1.0, 0.0)
            acc = torch.mean(score)
            self.log(
                "val_moc_loss", moc_loss, batch_size=agc_label.shape[0], sync_dist=True
            )
            self.log("val_moc_acc", acc, batch_size=agc_label.shape[0], sync_dist=True)

    def configure_optimizers(self) -> Any:
        return set_scheduler(self)


def pretrain(config_path: Path):
    config = yaml.full_load((cur_dir / "configs" / "config.model.yaml").open("r"))
    optimize_config = yaml.full_load(
        (cur_dir / "configs" / "config.optimize.yaml").open("r")
    )
    default_train_config = yaml.full_load(
        (cur_dir / "configs" / "config.pretrain.yaml").open("r")
    )

    base_config = {**config, **optimize_config, **default_train_config}

    with open(config_path.absolute(), "r") as f:
        user_config = yaml.full_load(f)

    if user_config.get("root_dir") is None:
        err(f"Please specify data root directory `root_dir`!")
        return
    if user_config.get("id_prop") is None:
        warn(f"Label data `id_prop` not specified, default is `vftopo`")
    if user_config.get("log_dir") is None:
        warn(f"Log directory not specified, defualt is `lightning_logs/pretrain`")

    base_config.update(user_config)

    config = base_config
    param(**config)

    title("START TO PRETRAIN")

    root_dir = Path(config["root_dir"])
    id_prop_path = root_dir / config["id_prop"]
    id_prop_dir = id_prop_path.parent
    df = pd.read_csv(id_prop_path)
    splits = ["train", "val"]
    if all(
        [
            (id_prop_dir / f"{id_prop_path.stem}.{split}.csv").exists()
            for split in splits
        ]
    ):
        print(f"Id prop file {id_prop_path.absolute()} has already been splitted")
        dfs = {
            split: pd.read_csv(
                id_prop_dir / f"{id_prop_path.stem}.{split}.csv", dtype={"cifid": str}
            )
            for split in splits
        }
        train_df = dfs["train"]
        val_df = dfs["val"]
    else:
        train_df, val_df = train_test_split(df, test_size=0.05, random_state=42)
        print(
            f"Id prop file {id_prop_path.absolute()} has not been splitted, split as 95:5, aka [{len(train_df)}:{len(val_df)}]"
        )
        train_df.to_csv(id_prop_dir / f"{id_prop_path.stem}.train.csv", index=False)
        val_df.to_csv(id_prop_dir / f"{id_prop_path.stem}.val.csv", index=False)
        dfs = {"train": train_df, "val": val_df, "all": df}

    modal_dir = root_dir / config["modal_folder"]

    # to get total number of topology
    datasets = {
        Dataset(
            df=dfs[split],
            modal_dir=modal_dir,
            nbr_fea_len=config["nbr_fea_len"],
            useBasis=config["useBasis"],
            useCharge=config["useCharge"],
            useMoc=config["useMoc"],
            img_size=config["img_size"]["lj"],
        )
        for split in (
            ["all"] + splits
        )  # order is important, because we need to include all topos in `topo2tid.json`
    }
    config["topo_num"] = dfs["all"].get_topo_num()
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=config["batch_size"],
            shuffle=split == "train",
            num_workers=8,
            collate_fn=datasets[split].collate,
        )
        for splits in splits
    }

    checkpoint_callback = ModelCheckpoint(
        monitor="val_agc_mae", mode="min", save_last=True
    )  # not important, we use last.ckpt for fine-tuning
    logger = TensorBoardLogger(save_dir=config["log_dir"], name="")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=config["epoch"],
        min_epochs=0,
        devices=config["device"],
        accelerator=config["accelerator"],
        strategy=config["strategy"],
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        precision=config["precision"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
    )
    model = SpbNetTrainer(config)
    trainer.fit(
        model=model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
        ckpt_path=config["resume"],  # to resume, default None
    )


@click.command()
@click.option(
    "--config-path", "-C", type=click.Path(exists=True, dir_okay=False, type=Path)
)
def pretrain_cli(config_path: Path):
    pretrain(config_path)


if __name__ == "__main__":
    pretrain_cli()
