from typing import Any, Optional
from functools import partial

# from pytorch_lightning.utilities.types import STEP_OUTPUT
# from model.crossformer import CrossFormer
from spbnet.modules.module import CrossFormer
from spbnet.modules.heads import RegressionHead, ClassificationHead, Pooler
from spbnet.modules.optimize import set_scheduler
from spbnet.modules import objectives
from torch import optim, nn
from data.dataset_pretrain import Dataset
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
from spbnet.utils.echo import err, title, start, end, param


cur_dir = Path(__file__).parent


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
        self.cross_entropy = nn.CrossEntropyLoss()

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
        # tc_loss = self.focal_loss(topo_pred, topo)
        tc_acc = self.cal_acc(topo, topo_pred)
        self.log("train_tc_loss", tc_loss, batch_size=topo.shape[0], sync_dist=True)
        self.log("train_tc_acc", tc_acc, batch_size=topo.shape[0], sync_dist=True)

        # atom grid
        potential_feat = feat[
            "potential_feat"
        ]  # [B, GRID / PathSize, GRID / PathSize, GRID / PathSize, hid_dim] aka [B, 10, 10, 10, 768]
        agc_pred = self.agc_head(potential_feat)
        agc_label: torch.Tensor = atomgrid
        patch_size = self.model_config["patch_size"]["lj"]
        # NOTE: Correct
        # agc_label = agc_label.transpose(-1, -3) # [b x y z] -> [b h w d]
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

        loss = (
            self.optimizer_config["vfp"] * vfp_mse
            + self.optimizer_config["tc"] * tc_loss
            + self.optimizer_config["agc"] * agc_mse
        )
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
        patch_size = self.model_config["patch_size"]["lj"]
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

    def cal_acc(self, y, pred):
        predicted_labels = torch.argmax(pred, dim=-1)
        correct_predictions = (predicted_labels == y).int()
        accuracy = correct_predictions.float().mean()
        return accuracy

    def configure_optimizers(self) -> Any:
        return set_scheduler(self)


def pretrain(config_path: Path):
    if type(config_path) is str:
        config_path = Path(config_path)
    if not config_path.exists():
        err(f"config_path: {config_path.absolute()} not exists!")

    with open((cur_dir / "config.pretrain.yaml").absolute(), "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_path.absolute(), "r") as f:
        user_config = yaml.load(f, Loader=yaml.FullLoader)

    if user_config.get("data_dir") is None:
        err(f"Please specify modal directory `data_dir`!")
        return
    if user_config.get("id_prop") is None:
        err(f"Please specify label data `id_prop`")
        return
    if user_config.get("log_dir") is None:
        err(f"Please specify log directory")
        return

    base_config.update(user_config)
    config = base_config
    param(**config)

    title("START TO PRETRAIN")

    df = pd.read_csv(config["id_prop"])
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    data_dir = Path(config["data_dir"])

    train_dataset = Dataset(train_df, data_dir, config["nbr_fea_len"])
    val_dataset = Dataset(val_df, data_dir, config["nbr_fea_len"])

    config["topo_num"] = train_dataset.get_topo_num()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_agc_mae", mode="min", save_last=True
    )
    logger = TensorBoardLogger(save_dir=config["log_dir"], name="")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        max_epochs=config["epoch"],
        min_epochs=0,
        devices=config["cuda"],
        accelerator=config["accelerator"],
        strategy=config["strategy"],
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        precision=config["precision"],
        log_every_n_steps=config["log_every_n_steps"],
        logger=logger,
    )
    model = CrossFormerTrainer(config)
    # ckpt = torch.load(
    #     "./lightning_logs/scratch/co2_2.5bar/shaper/checkpoints/epoch=146-step=13818.ckpt"
    # )
    # model.load_state_dict(ckpt["state_dict"], strict=False)
    trainer.fit(
        model=model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=8,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
            persistent_workers=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=8,
            collate_fn=partial(Dataset.collate, img_size=config["img_size"]),
            persistent_workers=True,
        ),
        # ckpt_path="./lightning_logs/version_2/checkpoints/last.ckpt",
    )
