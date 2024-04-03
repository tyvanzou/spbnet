import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cif", type=str)
args = parser.parse_args()
cifid = args.cif

# CO2
ckpt = "spbnet.180k"
# cifid = "cooperative"  # abb3976_data_s1

print(f"CIFID: {cifid}")

from .buildModalData import process

print("Start to prepare data")

process(cifid)

print("Prepare modal data end")

# %% [markdown]
# ## 获取注意力分数

# %%
import torch
import yaml
import torch.nn as nn

# from modules.module_reg import CrossFormer
from modules.module import CrossFormer
from modules.heads import RegressionHead, ClassificationHead, Pooler
from modules import objectives
import pytorch_lightning as pl
from pathlib import Path
from spbnet.utils.echo import start, end, title

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

    def pred_agc(self, batch) -> torch.Tensor:
        feat = self.model(batch)

        cls_feat = feat["cls_feat"]
        cls_feat = self.pooler(cls_feat)

        # atom grid
        potential_feat = feat[
            "potential_feat"
        ]  # [B, GRID / PathSize, GRID / PathSize, GRID / PathSize, hid_dim] aka [B, 10, 10, 10, 768]
        agc_pred = self.agc_head(potential_feat)

        return agc_pred


def predAgc(cifid: str, ckpt: str):
    title("START TO PRED AGC")

    start(f"Start to load weight from {ckpt}")

    ckpt = torch.load(
        ckpt,
        map_location="cpu",
    )
    state_dict = ckpt["state_dict"]
    with open((cur_dir / "./config.yaml").absolute(), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # config["visualize"] = True
    model = CrossFormerTrainer(config=config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    end(f"End to load weight from {ckpt}")

    # %%
    # print(model.potential_mapper.weight)
    # with torch.no_grad():
    #     model.potential_mapper.weight[:] = torch.ones((18)) * 2
    # model.potential_mapper.weight

    # %%
    from utils import get_grid_data, get_graph, collate
    import numpy as np
    from pathlib import Path

    start("Start to get pred agc")

    data_dir = Path("./modal")

    batch = []
    item = dict()
    item.update({"cifid": cifid, "target": 0})
    item.update(get_grid_data(data_dir, cifid))
    item.update(get_graph(data_dir, cifid))
    batch.append(item)
    batch = collate(batch)
    # cifiids = batch["cif_id"]  # list[str]
    # cifid = cifiids[0]
    # print(f"CIFID: {cifid}")
    agc_pred = model.pred_agc(batch)
    agc_pred = agc_pred.detach().cpu().reshape(-1)
    print(f"Agc getted: {agc_pred.shape}")
    np.save(f"./predagc/{cifid}.npy", agc_pred)
