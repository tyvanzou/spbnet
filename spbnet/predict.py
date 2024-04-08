from typing import Any, Optional
from functools import partial

# from pytorch_lightning.utilities.types import STEP_OUTPUT
# from model.crossformer import CrossFormer
from spbnet.modules.module import CrossFormer
from spbnet.modules.heads import RegressionHead, Pooler
from spbnet.modules.optimize import set_scheduler
from spbnet.modules import objectives
from torch import nn
from spbnet.data.dataset_predict import Dataset
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

    def on_test_epoch_start(self) -> None:
        self.test_items = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        feat = self.model(batch)
        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)
        reg = self.head(cls_feat)  # [batch_size, 1]
        pred = reg * self.std + self.mean

        cifid = batch["cifid"]
        for cifid_i, pred_i in zip(cifid, pred):
            self.test_items.append([cifid_i, pred_i.item()])

    def on_test_epoch_end(self) -> None:
        self.test_df = pd.DataFrame(
            self.test_items, columns=["cifid", "predict"]
        )
        pred = torch.tensor(self.test_df["predict"])
        self.test_df.to_csv(
            (Path(self.logger.log_dir) / "test_result.csv"),
            index=False,
        )
        return super().on_test_epoch_end()

    def configure_optimizers(self) -> Any:
        return set_scheduler(self)


def predict(config_path: str):
    base_config_path = cur_dir / "config.predict.yaml"
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
    if user_config.get("mean") is None:
        err(f"Please specify label data `mean`")
        return
    if user_config.get("std") is None:
        err(f"Please specify label data `std`")
        return

    # base_config.update(user_config)
    config = {**base_config, **user_config}

    # check
    title("PREDICT SPBNET")
    param(**config)

    # handle
    # task = config["task"]
    device = config["device"]
    id_prop_path = Path(config["id_prop"])
    ckpt_path = Path(config["ckpt"])  # TODO: CKPT
    data_dir = Path(config["data_dir"])
    log_dir = Path(config["log_dir"])

    # split train & val
    test_df = pd.read_csv(id_prop_path.absolute(), dtype={"cifid": str})
    test_dataset = Dataset(test_df, data_dir, config["nbr_fea_len"])

    # mean & std
    mean = config["mean"]
    std = config["std"]

    # train
    logger = TensorBoardLogger(save_dir=(log_dir).absolute(), name="")
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        min_epochs=0,
        devices=device,
        accelerator=config["accelerator"],
        strategy=config["strategy"],
        # callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        precision=config["precision"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
    )
    model = CrossFormerTrainer(config)

    trainer.test(
        model=model,
        dataloaders=DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=8,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
        ),
        ckpt_path=ckpt_path,
    )
