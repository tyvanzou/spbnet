from typing import Any, Optional
from functools import partial

# from pytorch_lightning.utilities.types import STEP_OUTPUT
# from model.crossformer import CrossFormer
from spbnet.modules.module import CrossFormer
from spbnet.modules.heads import RegressionHead, Pooler
from spbnet.modules.optimize import set_scheduler
from spbnet.modules import objectives
from torch import nn
from spbnet.data.dataset import Dataset
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
from spbnet.utils.echo import err, warn, param, title


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

        self.head = RegressionHead(model_config["hid_dim"])
        self.head.apply(objectives.init_weights)

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # mae
        self.min_mae = 1e5

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        lrs = self.lr_schedulers().get_last_lr()
        lr_avg = sum(lrs)
        lr_avg /= len(lrs)
        self.log("lr", lr_avg)

        # [batch_size, max_token_len]
        value = batch["target"]
        value = (value - self.mean) / self.std

        feat = self.model(batch)

        # regression
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        reg = self.head(cls_feat)  # [batch_size, 1]
        reg = reg.reshape(reg.shape[0])  # [batch_size]
        value = value.reshape(value.shape[0])  # [batch_size]
        mse = self.mse_loss(reg, value)
        mae = self.mae_loss(reg, value)
        self.log("train_mse", mse, batch_size=value.shape[0], sync_dist=True)
        self.log("train_mae", mae, batch_size=value.shape[0], sync_dist=True)

        # raw
        target = batch["target"]
        pred = reg * self.std + self.mean
        mse_raw = torch.mean((target - pred) ** 2)
        mae_raw = torch.mean(torch.abs(target - pred))
        r2 = r2_score(target, pred)
        self.log(
            "train_mse_raw",
            mse_raw,
            batch_size=value.shape[0],
            sync_dist=True,
        )
        self.log("train_mae_raw", mae_raw, batch_size=value.shape[0], sync_dist=True)
        self.log("train_r2", r2, batch_size=value.shape[0], sync_dist=True)

        # norm_mse = self.mse_loss(reg, norm_value)
        # loss = norm_mse
        loss = mse
        # loss = bert_loss
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_items = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        # [batch_size, max_token_len]
        target = batch["target"]

        feat = self.model(batch)
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        reg = self.head(cls_feat)  # [batch_size, 1]
        pred = reg * self.std + self.mean

        cifid = batch["cifid"]
        for cifid_i, target_i, pred_i in zip(cifid, target, pred):
            self.validation_items.append([cifid_i, target_i.item(), pred_i.item()])

    def on_validation_epoch_end(self) -> None:
        self.validation_df = pd.DataFrame(
            self.validation_items, columns=["cifid", "target", "predict"]
        )
        pred = torch.tensor(self.validation_df["predict"])
        target = torch.tensor(self.validation_df["target"])
        mae = torch.mean(torch.abs((pred - target)))
        mse = torch.mean((pred - target) ** 2)
        r2 = r2_score(target, pred)
        self.log("val_mae", mae)
        self.log("val_mse", mse)
        self.log("val_r2", r2)
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.test_items = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        # [batch_size, max_token_len]
        target = batch["target"]

        feat = self.model(batch)
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        reg = self.head(cls_feat)  # [batch_size, 1]
        pred = reg * self.std + self.mean

        cifid = batch["cifid"]
        for cifid_i, target_i, pred_i in zip(cifid, target, pred):
            self.test_items.append([cifid_i, target_i.item(), pred_i.item()])

    def on_test_epoch_end(self) -> None:
        self.test_df = pd.DataFrame(
            self.test_items, columns=["cifid", "target", "predict"]
        )
        pred = torch.tensor(self.test_df["predict"])
        target = torch.tensor(self.test_df["target"])
        mae = torch.mean(torch.abs((pred - target)))
        mse = torch.mean((pred - target) ** 2)
        r2 = r2_score(target, pred)
        self.log("test_mae", mae)
        self.log("test_mse", mse)
        self.log("test_r2", r2)
        # print(f"TEST MAE: {mae.item()}, R2: {r2.item()}")
        self.test_df.to_csv(
            (Path(self.logger.log_dir) / "test_result.csv"),
            index=False,
        )
        return super().on_test_epoch_end()

    def configure_optimizers(self) -> Any:
        return set_scheduler(self)


def finetune(config_path: str):
    base_config_path = cur_dir / "config.finetune.yaml"
    with open(base_config_path, "r") as f:
        base_config: dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_path, "r") as f:
        user_config: dict = yaml.load(f, Loader=yaml.FullLoader)

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

    # base_config.update(user_config)
    config = {**base_config, **user_config}

    # check
    title("FINETUNE SPBNET")
    param(**config)

    # handle
    task = config["task"]
    device = config["device"]
    id_prop_path = Path(config["id_prop"])
    ckpt_path = Path(config["ckpt"])  # TODO: CKPT
    data_dir = Path(config["data_dir"])
    log_dir = Path(config["log_dir"])

    # split train & val
    id_prop_dir = id_prop_path.parent
    train_df = pd.read_csv(
        id_prop_dir / f"{id_prop_path.stem}.train.csv", dtype={"cifid": str}
    )
    val_df = pd.read_csv(
        id_prop_dir / f"{id_prop_path.stem}.validate.csv", dtype={"cifid": str}
    )
    test_df = pd.read_csv(
        id_prop_dir / f"{id_prop_path.stem}.test.csv", dtype={"cifid": str}
    )
    train_dataset = Dataset(train_df, data_dir, config["nbr_fea_len"], task)
    val_dataset = Dataset(val_df, data_dir, config["nbr_fea_len"], task)
    test_dataset = Dataset(test_df, data_dir, config["nbr_fea_len"], task)

    # id_prop
    id_prop_df = train_df.copy()
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
        model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=8,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
        ),
    )

    trainer.test(
        dataloaders=DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=8,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
        ),
        ckpt_path="best",
    )
