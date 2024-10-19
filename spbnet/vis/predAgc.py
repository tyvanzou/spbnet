from pathlib import Path
import click

import numpy as np
import torch
import yaml
import torch.nn as nn
import pytorch_lightning as pl

from ..modules.module import SpbNet
from ..utils.echo import start, end, title, warn
from .utils import get_grid_data, get_graph, collate
from .buildModal import buildModal


cur_dir = Path(__file__).parent
root_dir = cur_dir.parent


class SpbNetTrainer(pl.LightningModule):
    def __init__(self, config: dict):
        super(SpbNetTrainer, self).__init__()

        self.config = config

        self.model = SpbNet(config)

        # atom grid classify
        self.agc_head = nn.Linear(config["hid_dim"], 1)

    def pred_agc(self, batch) -> torch.Tensor:
        feat = self.model(batch)

        # atom grid
        potential_feat = feat[
            "potential_feat"
        ]  # [B, GRID / PathSize, GRID / PathSize, GRID / PathSize, hid_dim] aka [B, 10, 10, 10, 768]
        agc_pred = self.agc_head(potential_feat)

        return agc_pred


@click.command()
@click.option(
    "--cif-path", "-C", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--modal-dir", "-M", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--ckpt', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--out-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def predAgc(cif_path: Path, modal_dir: Path, ckpt: Path, out_dir: Path):
    title("START TO PRED AGC")

    warn("NOTE: This code is simply used to draw the graph in the paper. Currently the code has not been adjusted to be applied to models using charge, grad, and moc. You can use the ckpt file provided `spbnet.160k.ckpt`, `spbnet.1m.ckpt` or modify the code.")

    start("Start to prepare data")

    buildModal(cif_path, modal_dir)

    ckpt_path = ckpt
    start(f"Start to load weight from {ckpt_path}")

    ckpt = torch.load(
        ckpt,
        map_location="cpu",
    )
    state_dict = ckpt["state_dict"]

    # hparams = ckpt['hyper_parameters']
    # config = hparams['config']
    # print(config)

    # NOTE: only default config of model can be used
    with (root_dir / 'configs' / 'config.model.yaml').open('r') as f:
        config = yaml.full_load(f)

    # config["visualize"] = True
    model = SpbNetTrainer(config=config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    end(f"End to load weight from {ckpt_path}")

    start("Start to predict atom number")

    data_dir = modal_dir

    batch = []
    item = dict()
    cifid = cif_path.stem
    item.update({"cifid": cifid})
    item.update(get_grid_data(data_dir, cifid))
    item.update(get_graph(data_dir, cifid))
    batch.append(item)
    batch = collate(batch)

    agc_pred = model.pred_agc(batch)
    agc_pred = agc_pred.detach().cpu().reshape(-1)
    print(f"Agc get: {agc_pred.shape}")

    out_dir.mkdir(exist_ok=True, parents=True)

    np.save(out_dir / f"{cifid}.npy", agc_pred)


if __name__ == '__main__':
    predAgc()
