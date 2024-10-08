from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import yaml
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch.optim as optim
from functools import partial

from datamodule import CifGraphDataset
from torch_geometric.nn.models import ViSNet, DimeNetPlusPlus
from spherenet import SphereNet

from optimize import set_scheduler
from logs import log

torch.set_float32_matmul_precision("medium")


class GNNTrainer(pl.LightningModule):
    def __init__(self, config):
        super(GNNTrainer, self).__init__()
        self.save_hyperparameters()

        self.config = config
        self.gnn = self.config["gnn"]

        self.mean = config["mean"]
        self.std = config["std"]

        if self.config["gnn"] == "visnet":
            self.visnet = ViSNet(**config["visnet"])
        elif self.config["gnn"] == "dimenetpp":
            self.dimenetpp = DimeNetPlusPlus(**config["dimenet"])
        elif self.config["gnn"] == "spherenet":
            self.spherenet = SphereNet(**config["spherenet"])

        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        geodata = batch["geometric"]
        if self.gnn == "visnet":
            pred = self.visnet(geodata["x"], geodata["pos"], geodata["batch"])[0]
            # x, v = self.visnet.representation_model(geodata['x'], geodata['pos'], geodata['batch'])[0]
        elif self.gnn == "dimenet":
            pred = self.dimenetpp(geodata["x"], geodata["pos"], geodata["batch"])
        elif self.gnn == "spherenet":
            pred = self.spherenet(geodata["x"], geodata["pos"], geodata["batch"])[0]

        return pred

    def training_step(self, batch, batch_idx):
        lrs = self.lr_schedulers().get_last_lr()
        lr_avg = sum(lrs)
        lr_avg /= len(lrs)
        self.log("lr", lr_avg)

        pred = self.forward(batch)
        label = batch["target"]
        batch_size = label.shape[0]
        label = label.reshape(batch_size)
        pred = pred.reshape(batch_size)

        label_norm = (label - self.mean) / self.std
        mse = self.mse_loss(label_norm, pred)
        mae = self.mae_loss(label_norm, pred)
        pred_denorm = pred * self.std + self.mean

        log(
            self, target=label, pred=pred_denorm, prefix="train", batch_size=batch_size
        )

        return mse

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        label = batch["target"]
        batch_size = label.shape[0]
        label = label.reshape(batch_size)
        pred = pred.reshape(batch_size)
        # print(pred, self.mean, self.std)
        pred_denorm = pred * self.std + self.mean

        log(self, target=label, pred=pred_denorm, prefix="val", batch_size=batch_size)

    def on_test_epoch_start(self):
        self.test_items = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        pred = self.forward(batch)

        label = batch["target"]
        batch_size = label.shape[0]
        pred_denorm = pred * self.std + self.mean
        label = label.reshape(batch_size)
        pred = pred.reshape(batch_size)

        cifid = batch["cifids"]
        assert len(cifid) == batch["target"].shape[0]
        assert len(cifid) == pred_denorm.shape[0]
        for cifid_i, target_i, pred_i in zip(cifid, batch["target"], pred_denorm):
            self.test_items.append([cifid_i, target_i.item(), pred_i.item()])

    def on_test_epoch_end(self):
        self.test_df = pd.DataFrame(
            self.test_items, columns=["cifid", "target", "predict"]
        )
        pred = torch.tensor(self.test_df["predict"])
        target = torch.tensor(self.test_df["target"])

        log(self, target=label, pred=pred_denorm, prefix="test", batch_size=batch_size)

        self.test_df.to_csv(
            (Path(self.logger.log_dir) / "test_result.csv"),
            index=False,
        )

        return super().on_test_epoch_end()

    def configure_optimizers(self):
        # optimizer = optim.AdamW(
        #     self.parameters(),
        #     lr=0.0001,
        # )
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return set_scheduler(self)


def main():
    with open("./config.yaml") as f:
        config = yaml.full_load(f)

    log_dir = Path(config["log_dir"])
    task = config["task"]
    logger = TensorBoardLogger(save_dir=(log_dir / task).absolute(), name="")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        precision=config["precision"],
        strategy=config["strategy"],
        devices=config["devices"],
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=config["accelerator"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
    )

    root_dir = Path(config["root_dir"])
    id_prop_path = root_dir / f"{config['id_prop']}.csv"

    df = pd.read_csv(id_prop_path)
    df = df.dropna(subset=[config["task"]])
    targets = df[config["task"]]
    mean, std = targets.mean(), targets.std()
    config["mean"] = float(mean)
    config["std"] = float(std)

    # split train & val
    id_prop_dir = id_prop_path.parent
    splits = ["train", "val", "test"]
    if all(
        [
            (id_prop_dir / f"{id_prop_path.stem}.{split}.csv").exists()
            for split in splits
        ]
    ):
        dfs = {
            split: pd.read_csv(
                id_prop_dir / f"{id_prop_path.stem}.{split}.csv", dtype={"cifid": str}
            )
            for split in splits
        }
    else:
        df = pd.read_csv(id_prop_path)
        train_df, val_df = train_test_split(df, test_size=1 / 7, random_state=42)
        train_df, test_df = train_test_split(train_df, test_size=1 / 6, random_state=42)
        train_df.to_csv(id_prop_dir / f"{id_prop_path.stem}.train.csv", index=False)
        val_df.to_csv(id_prop_dir / f"{id_prop_path.stem}.val.csv", index=False)
        test_df.to_csv(id_prop_dir / f"{id_prop_path.stem}.test.csv", index=False)
        dfs = {"train": train_df, "val": val_df, "test": test_df}
    datasets = {
        split: CifGraphDataset(root_dir, dfs[split], target=config["task"])
        for split in splits
    }
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=config["batch_size"],
            collate_fn=partial(CifGraphDataset.collate, isFinetune=True),
            num_workers=8,
        )
        for split in splits
    }

    model = GNNTrainer(config=config)

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


if __name__ == "__main__":
    main()
