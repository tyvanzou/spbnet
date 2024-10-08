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

from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import R2Score

from .datamodule.dataset import FinetuneDataset as Dataset
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
            self.acc = MulticlassAccuracy(num_classes=self.cls_num)

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
            reg = pred.reshape(batch_size)  # [batch_size]

            # raw
            reg = reg * self.std + self.mean

            cifid = batch["cifid"]
            assert len(cifid) == reg.shape[0]
            for cifid_i, pred_i in zip(cifid, reg):
                self.test_items.append([cifid_i, pred_i.item()])

        elif self.task_type == "classification":
            predicted_labels = (
                torch.argmax(pred, dim=-1).reshape(-1).detach().cpu().numpy().tolist()
            )
            predicted_labels = [self.id_cls_map[l] for l in predicted_labels]

            cifid = batch["cifid"]
            for cifid_i, pred_i in zip(cifid, predicted_labels):
                self.test_items.append([cifid_i, pred_i])

    def on_test_epoch_end(self) -> None:
        self.test_df = pd.DataFrame(self.test_items, columns=["cifid", "predict"])
        self.test_df.to_csv(
            (Path(self.logger.log_dir) / "test_result.csv"),
            index=False,
        )
        return super().on_test_epoch_end()

    def configure_optimizers(self) -> Any:
        return set_scheduler(self)


def predict(config_path: Path):
    torch.set_float32_matmul_precision("medium")

    model_config = yaml.full_load((cur_dir / "configs" / "config.model.yaml").open("r"))
    optimize_config = yaml.full_load(
        (cur_dir / "configs" / "config.optimize.yaml").open("r")
    )
    default_train_config = yaml.full_load(
        (cur_dir / "configs" / "config.finetune.yaml").open("r")
    )

    with config_path.open("r") as f:
        user_config: dict = yaml.full_load(f)

    if user_config.get("root_dir") is None:
        err(f"Please specify root directory `root_dir`")
        return
    if user_config.get("ckpt") is None:
        err(f"Please specify checkpoint `ckpt`")
        return
    if user_config.get("task_type") is None:
        warn(f"Task_type not specified, use default `regression`")
    if user_config.get("id_prop") is None:
        warn(f"Label data `id_prop` not specified, use default `benchmark.test`")
    if user_config.get("log_dir") is None:
        warn(f"Log directory not specified, use default `./lightning_logs/predict`")

    config = {**model_config, **optimize_config, **default_train_config, **user_config}

    # check
    title("PREDICT SPBNET")
    param(**config)

    # handle
    root_dir = Path(config["root_dir"])
    id_prop = config["id_prop"]
    id_prop_path = root_dir / f"{id_prop}.csv"
    modal_dir = root_dir / config["modal_folder"]
    device = config["device"]
    log_dir = Path(config["log_dir"])

    ckpt_path = Path(config["ckpt"])
    ckpt = torch.load(ckpt_path, weights_only=True)
    hparams = ckpt["hyper_parameters"]["config"]

    # id_prop
    task_type = hparams["task_type"]
    config["task_type"] = task_type

    # split train & val
    df = pd.read_csv(str(id_prop_path))

    dataset = Dataset(
        df=df,
        modal_dir=modal_dir,
        nbr_fea_len=config["nbr_fea_len"],
        task_type=task_type,
        useBasis=config["useBasis"],
        useCharge=config["useCharge"],
        img_size=config["img_size"]["lj"],
        isPredict=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )

    # regression
    if task_type == "regression":
        # mean & std
        config["mean"] = hparams["mean"]
        config["std"] = hparams["std"]
    elif task_type == "classification":
        config["cls_num"] = hparams["cls_num"]
        config["cls_id_map"] = hparams["cls_id_map"]
        config["id_cls_map"] = hparams["id_cls_map"]

    # train
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae" if task_type == "regression" else "val_acc",
        mode="min" if task_type == "regression" else "max",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=(log_dir).absolute(), name="")
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

    loadret = model.load_state_dict(ckpt["state_dict"], strict=True)
    print("Load return", loadret)

    trainer.test(
        model=model,
        dataloaders=loader,
    )


@click.command()
@click.option(
    "--config-path", "-C", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def predict_cli(config_path: Path):
    predict(config_path)


if __name__ == "__main__":
    predict_cli()
