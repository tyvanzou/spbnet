import click

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
from einops import rearrange
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd

from ..utils import *

def distance(pos1, pos2):
    def get_pos(a):
        az = a % 10
        a = a // 10
        ay = a % 10
        a = a // 10
        ax = a % 10
        a = a // 10
        return ax, ay, az

    ss = np.sum((np.array(get_pos(pos1)) - np.array(get_pos(pos2))) ** 2)
    dis = np.sqrt(ss)

    # if ss == 25 or ss == 50 or ss == 49:
    #     return dis
    # else:
    #     return 0

    return dis

@click.command()
@click.option('--self-attns', '-S', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--out-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def main(self_attns: Path, out_dir: Path):
    start("Start to load self attns")
    show = np.load(self_attns)
    print("Load end, attn shape: ", show.shape)
    # show = np.mean(show, axis=0)

    # show = rearrange(show.reshape((10, 10, 10, -1)), "z y x a -> x y z a")
    show = rearrange(show.reshape((-1, 10, 10, 10, 1000)), "b z y x a -> b x y z a")
    indices = np.random.randint(0, show.shape[0], 1000)
    show = show[indices]

    # Heatmap
    show = show.reshape((-1, 1000, 1000))
    show = np.mean(show, axis=0)
    # show = show[425:475, 425:475]
    start, end = 400, 501
    show = show[start:end, start:end]
    # show = show[500]

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", list(zip([0, 1], ["#faf8fc", "#8387a1"]))
    )

    vmax = np.max(show) * 0.4
    # plt.figure(figsize=(7, 6))
    plt.imshow(show, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    plt.xlim([0, end - start])
    plt.ylim([0, end - start])
    plt.xticks(
        ticks=[i for i in range(0, end - start, 20)],
        labels=[i for i in range(start, end, 20)],
    )
    plt.yticks(
        ticks=[i for i in range(0, end - start, 20)],
        labels=[i for i in range(start, end, 20)],
    )
    # sns.kdeplot(
    #     show,
    #     cmap="mako",
    #     fill=True,
    #     thresh=0,
    #     levels=100,
    # )
    plt.colorbar()
    plt.title("Heatmap of self attention", fontsize=18, pad=20)
    plt.ylabel("Patch of potential", fontsize=16, labelpad=10)
    plt.xlabel("Patch of potential", fontsize=16, labelpad=10)

    out_dir.mkdir(exists_ok=True, parents=True)
    plt.savefig(
        out_dir / "self_attn.finegrained.png", dpi=300, format="png", bbox_inches="tight"
    )

    end(f"File saved in {out_dir / 'self_attn.finegrained.png'}")
    # plt.savefig("./self_attn_center.png", dpi=300, format="png", bbox_inches="tight")
    # plt.savefig("./self_attn.png", dpi=300, format="png", bbox_inches="tight")
