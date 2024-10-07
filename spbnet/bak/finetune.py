from typing import Any, Optional
from functools import partial
import click

from torch import nn
import pytorch_lightning as pl
import pandas as pd
import yaml
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from pathlib import Path
from multiprocessing import Process
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from torchmetrics import Accuracy

from .datamodule.dataset import Dataset
from .modules.module import SpbNet
from .modules.heads import RegressionHead, Pooler, ClassificationHead
from .modules.optimize import set_scheduler
from .modules import objectives
from .utils.echo import err, warn, param, title


cur_dir = Path(__file__).parent


class SpbNetTrainer(pl.LightningModule):
    def __init__(self, config: dict):
        super(SpbNetTrainer, self).__init__()
        self.save_hyperparameters()

        self.config = config
        self.model_config = config
        self.optimizer_config = config

        model_config = self.model_config
        self.model = SpbNet(model_config)
        # print(self.model)
        # self.model = AtomFormer(config)

        # pooler
        self.pooler = Pooler(model_config["hid_dim"])
        self.pooler.apply(objectives.init_weights)

        self.task_type = config["task_type"]

        if config["task_type"] == "regression":
            self.mean = config["mean"]
            self.std = config["std"]

            self.head = RegressionHead(model_config["hid_dim"])
            self.head.apply(objectives.init_weights)
        elif config["task_type"] == "classification":
            self.cls_num = config["cls_num"]
            self.cls_id_map = config["cls_id_map"]
            self.id_cls_map = config["id_cls_map"]

            self.head = ClassificationHead(model_config["hid_dim"], self.cls_num)
            self.head.apply(objectives.init_weights)

        if config["useTopoHelp"]:
            self.topo_head = ClassificationHead(
                model_config["hid_dim"], config["topo_num"]
            )
            self.topo_head.apply(objectives.init_weights)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.acc = Accuracy()

        # mae
        self.min_mae = 1e5

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        lrs = self.lr_schedulers().get_last_lr()
        lr_avg = sum(lrs)
        lr_avg /= len(lrs)
        self.log("lr", lr_avg)

        feat = self.model(batch)
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        pred = self.head(
            cls_feat
        )  # [batch_size, 1] for regression and [B, cls_num] for classification
        batch_size = pred.shape[0]
        device = pred.device

        if self.task_type == "regression":
            # [batch_size, max_token_len]
            value = batch["target"]
            value = (value - self.mean) / self.std

            reg = pred.reshape(batch_size)  # [batch_size]
            value = value.reshape(batch_size)  # [batch_size]
            mse = self.mse_loss(reg, value)
            mae = self.mae_loss(reg, value)
            self.log("train_mse", mse, batch_size=batch_size, sync_dist=True)
            self.log("train_mae", mae, batch_size=batch_size, sync_dist=True)

            # raw
            target = batch["target"]
            pred = reg * self.std + self.mean
            mse_raw = torch.mean((target - pred) ** 2)
            mae_raw = torch.mean(torch.abs(target - pred))
            r2 = r2_score(target, pred)
            self.log(
                "train_mse_raw",
                mse_raw,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log("train_mae_raw", mae_raw, batch_size=batch_size, sync_dist=True)
            self.log("train_r2", r2, batch_size=batch_size, sync_dist=True)

            loss = mse

        elif self.task_type == "classification":
            value = batch["target"]  # List[str]
            cls_ids = [self.cls_id_map[t] for t in value]
            cls_ids = torch.tensor(cls_ids, dtype=torch.long).to(device)  # [B, ]
            loss = self.cross_entropy(pred, cls_ids)
            acc = self.cal_acc(cls_ids, pred)
            self.log("train_cross_entropy", loss, batch_size=batch_size, sync_dist=True)
            self.log("train_acc", acc, batch_size=batch_size, sync_dist=True)

        if self.config["useTopoHelp"]:
            topo = batch["topo"]  # tensor [B]
            topo_pred = self.topo_head(cls_feat)
            topo_loss = self.cross_entropy(topo_pred, topo)
            acc = self.cal_acc(topo, topo_pred)
            self.log(
                "train_topo_entropy", topo_loss, batch_size=batch_size, sync_dist=True
            )
            self.log("train_topo_acc", acc, batch_size=batch_size, sync_dist=True)

            loss += topo_loss * self.config["topo_loss_weight"]

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        feat = self.model(batch)
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        pred = self.head(
            cls_feat
        )  # [batch_size, 1] for regression and [B, cls_num] for classification
        batch_size = pred.shape[0]
        device = pred.device

        if self.task_type == "regression":
            # [batch_size, max_token_len]
            value = batch["target"]
            value = (value - self.mean) / self.std

            reg = pred.reshape(batch_size)  # [batch_size]
            value = value.reshape(batch_size)  # [batch_size]
            target = batch["target"]
            pred = reg * self.std + self.mean
            mse_raw = torch.mean((target - pred) ** 2)
            mae_raw = torch.mean(torch.abs(target - pred))
            r2 = r2_score(target, pred)
            self.log(
                "val_mse",
                mse_raw,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log("val_mae", mae_raw, batch_size=batch_size, sync_dist=True)
            self.log("val_r2", r2, batch_size=batch_size, sync_dist=True)

        elif self.task_type == "classification":
            value = batch["target"]  # List[str]
            cls_ids = [self.cls_id_map[t] for t in value]
            cls_ids = torch.tensor(cls_ids, dtype=torch.long).to(device)  # [B, ]
            loss = self.cross_entropy(pred, cls_ids)
            acc = self.cal_acc(cls_ids, pred)
            self.log("val_cross_entropy", loss, batch_size=batch_size, sync_dist=True)
            self.log("val_acc", acc, batch_size=batch_size, sync_dist=True)

        if self.config["useTopoHelp"]:
            topo = batch["topo"]  # tensor [B]
            topo_pred = self.topo_head(cls_feat)
            topo_loss = self.cross_entropy(topo_pred, topo)
            acc = self.cal_acc(topo, topo_pred)
            self.log(
                "val_topo_entropy", topo_loss, batch_size=batch_size, sync_dist=True
            )
            self.log("val_topo_acc", acc, batch_size=batch_size, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        self.test_items = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        feat = self.model(batch)
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        pred = self.head(
            cls_feat
        )  # [batch_size, 1] for regression and [B, cls_num] for classification
        batch_size = pred.shape[0]

        if self.task_type == "regression":
            # [batch_size, max_token_len]
            value = batch["target"]
            value = (value - self.mean) / self.std

            reg = pred.reshape(batch_size)  # [batch_size]
            value = value.reshape(batch_size)  # [batch_size]

            # raw
            target = batch["target"]
            reg = reg * self.std + self.mean
            mse_raw = torch.mean((target - reg) ** 2)
            mae_raw = torch.mean(torch.abs(target - reg))
            r2 = r2_score(target, reg)
            self.log(
                "test_mse",
                mse_raw,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log("test_mae", mae_raw, batch_size=batch_size, sync_dist=True)
            self.log("test_r2", r2, batch_size=batch_size, sync_dist=True)

            cifid = batch["cifid"]
            assert len(cifid) == batch["target"].shape[0]
            assert len(cifid) == reg.shape[0]
            for cifid_i, target_i, pred_i in zip(cifid, batch["target"], reg):
                self.test_items.append([cifid_i, target_i.item(), pred_i.item()])

        elif self.task_type == "classification":
            value = batch["target"]  # List[str]
            predicted_labels = (
                torch.argmax(pred, dim=-1).reshape(-1).detach().cpu().numpy().tolist()
            )
            predicted_labels = [self.id_cls_map[l] for l in predicted_labels]
            cls_ids = [self.cls_id_map[t] for t in value]

            cifid = batch["cifid"]
            for cifid_i, target_i, pred_i in zip(cifid, batch["target"], cls_ids):
                self.test_items.append([cifid_i, target_i, pred_i])

    def on_test_epoch_end(self) -> None:
        self.test_df = pd.DataFrame(
            self.test_items, columns=["cifid", "target", "predict"]
        )
        pred = torch.tensor(self.test_df["predict"])
        target = torch.tensor(self.test_df["target"])

        if self.task_type == "regression":
            mae = torch.mean(torch.abs((pred - target)))
            mse = torch.mean((pred - target) ** 2)
            r2 = r2_score(target, pred)
            self.log("test_mae", mae)
            self.log("test_mse", mse)
            self.log("test_r2", r2)
        elif self.task_type == "classification":
            acc_list = [
                1 if pred_i == target_i else 0
                for i, (target_i, pred_i) in enumerate(zip(target, pred))
            ]
            acc = torch.tensor(acc_list).sum().item() / len(acc_list)
            self.log("test_acc", acc, sync_dist=True)
        # print(f"TEST MAE: {mae.item()}, R2: {r2.item()}")
        self.test_df.to_csv(
            (Path(self.logger.log_dir) / "test_result.csv"),
            index=False,
        )
        return super().on_test_epoch_end()

    def cal_acc(self, y, pred):
        predicted_labels = torch.argmax(pred, dim=-1)
        correct_predictions = (predicted_labels == y).int()
        accuracy = correct_predictions.float().mean()
        return accuracy

    def configure_optimizers(self) -> Any:
        return set_scheduler(self)


def finetune(config_path: str):
    config = yaml.load((cur_dir / "configs" / "config.model.yaml").open("r"))
    optimize_config = yaml.load(
        (cur_dir / "configs" / "config.optimize.yaml").open("r")
    )
    default_train_config = yaml.load(
        (cur_dir / "configs" / "config.finetune.yaml").open("r")
    )

    with open(config_path, "r") as f:
        user_config: dict = yaml.load(f, Loader=yaml.FullLoader)

    if user_config.get("root_dir") is None:
        err(f"Please specify root directory `root_dir`")
        return
    if user_config.get("task") is None:
        err(f"Please specify task `task`")
        return
    if user_config.get("task_type") is None:
        warn(f"Task_type not specified, use default `regression`")
    if user_config.get("id_prop") is None:
        warn(f"Label data `id_prop` not specified, use default `benchmark`")
    if user_config.get("log_dir") is None:
        warn(f"Log directory not specified, use default `./lightning_logs/finetune`")
        return

    # base_config.update(user_config)
    config = {**base_config, **user_config}

    # check
    title("FINETUNE SPBNET")
    param(**config)

    # handle
    root_dir = Path(config["root_dir"])
    task = config["task"]
    task_type = config["task_type"]
    id_prop = config["id_prop"]
    id_prop_path = root_dir / id_prop
    ckpt_path = Path(config["ckpt"])
    modal_dir = root_dir / config["modal_folder"]
    device = config["device"]
    log_dir = Path(config["log_dir"])

    # split train & val
    id_prop_dir = id_prop_path.parent
    splits = ["train", "val", "test"]
    if all([(id_prop_dir / f"{id_prop}.{split}.csv").exists() for split in splits]):
        print(f"id_prop file has already been splitted.")
        dfs = {
            split: pd.read_csv(
                id_prop_dir / f"{id_prop}.{split}.csv", dtype={"cifid": str}
            )
            for split in splits
        }
    else:
        df = pd.read_csv(config["id_prop"])
        train_df, val_df = train_test_split(df, test_size=1 / 7, random_state=42)
        train_df, test_df = train_test_split(train_df, test_size=1 / 6, random_state=42)
        train_df.to_csv(id_prop_dir / f"{id_prop}.train.csv", index=False)
        val_df.to_csv(id_prop_dir / f"{id_prop}.val.csv", index=False)
        test_df.to_csv(id_prop_dir / f"{id_prop}.test.csv", index=False)
        print(
            f'id_prop file has already been splitted, split {str(id_prop_dir / f"{id_prop}.{split}.csv")} into 5:1:1, aka {len(train_df)}:{len(val_df)}:{len(test_df)}'
        )
        dfs = {"train": train_df, "val": val_df, "test": test_df}
    datasets = {
        split: Dataset(
            df=dfs[split],
            modal_dir=modal_dir,
            nbr_fea_len=config["nbr_fea_len"],
            task=task,
            task_type=task_type,
            useBasis=config["useBasis"],
            useCharge=config["useCharge"],
            img_size=config["img_size"]["lj"],
            isPredict=False,
        )
        for split in splits
    }
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=config["batch_size"],
            shuffle=split == "train",
            num_workers=8,
            collate_fn=datasets[split].collate_fn,
        )
        for split in splits
    }

    # id_prop
    id_prop_df = dfs["train"].copy()
    filter_id_prop_df = id_prop_df.dropna(subset=[task])

    # regression
    if config["task_type"] == "regression":
        # mean & std
        mean = None
        std = None
        mean = filter_id_prop_df[task].mean()
        std = filter_id_prop_df[task].std()
        config["mean"] = float(mean)
        config["std"] = float(std)
    elif config["task_type"] == "classification":
        cls_series = filter_id_prop_df[task].value_counts().index.tolist()
        cls_num = len(cls_series)
        cls_id_map = {clsname: i for i, clsname in enumerate(cls_series)}
        id_cls_map = {i: clsname for i, clsname in enumerate(cls_series)}
        config["cls_num"] = cls_num
        config["cls_id_map"] = cls_id_map
        config["id_cls_map"] = id_cls_map

    # train
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae" if task_type == "regression" else "val_acc",
        mode="min" if task_type == "regression" else "max",
        save_last=True,
    )
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
    model = SpbNetTrainer(config)
    if not config["ckpt"] == "scratch":
        ckpt = torch.load(ckpt_path)
        loadret = model.load_state_dict(ckpt["state_dict"], strict=False)
        print("Load return", loadret)
    trainer.fit(
        model=model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )

    trainer.test(
        dataloaders=loaders["test"],
        ckpt_path="best",
    )


@click.command()
@click.option("--config-path", type=str)
def finetune_cli(config_path: str):
    finetune(config_path)


if __name__ == "__main__":
    finetune_cli()
