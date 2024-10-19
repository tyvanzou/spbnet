import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
import click

from ..utils.echo import start, end

@click.group()
def cli():
    pass


@cli.command()
@click.option("--agc-labels", "-L", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--agc-preds", "-P", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def boxplot(agc_labels: Path, agc_preds: Path, output_dir: Path):
    start("Load agc labels and preds")

    agc_labels = np.load(agc_labels)
    agc_preds = np.load(agc_preds)

    start(
        f"Load end: label({agc_labels.shape}), pred({agc_preds.shape}), start to draw box plot"
    )

    print("R2", r2_score(agc_labels, agc_preds))
    print("pearsonr", pearsonr(agc_labels, agc_preds)[0])

    idxes = np.where(agc_labels > 0)[0]
    agc_labels = agc_labels[idxes]
    agc_preds = agc_preds[idxes]

    idxes = np.where(agc_labels < 10)[0]
    agc_labels = agc_labels[idxes]
    agc_preds = agc_preds[idxes]

    data = {"agc_labels": agc_labels, "agc_preds": agc_preds}

    data = pd.DataFrame(data)

    mpl.rcParams["font.size"] = 14

    fig = plt.figure(figsize=(9, 7))

    subfig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        x="agc_labels",
        y="agc_preds",
        data=data,
        showfliers=False,
        ax=ax,
        color="#d8bfca",
    )
    sns.despine(offset=10, trim=True)
    plt.xlabel(
        "Actual atom number",
        labelpad=10,
        fontdict={"size": 18, "family": "Arial", "color": "black"},
    )
    plt.ylabel(
        "Predicted atom number",
        labelpad=10,
        fontdict={"size": 18, "family": "Arial", "color": "black"},
    )
    plt.title(
        "Boxplot of predicted atom number",
        pad=20,
        fontdict={"size": 18, "family": "Arial", "color": "black"},
    )

    # plt.savefig("./eps/agc.boxplot.eps", format="eps", bbox_inches="tight")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "agc.boxplot.png", format="png", dpi=300, bbox_inches="tight")

    end(f"draw end, file saved in {output_dir / 'agc.boxplot.png'}")


@cli.command()
@click.option("--agc-labels", "-L", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--agc-preds", "-P", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def jointplot(agc_labels: Path, agc_preds: Path, output_dir: Path):
    start("Load agc labels and preds")

    agc_labels = np.load(agc_labels)
    agc_preds = np.load(agc_preds)

    start(
        f"Load end: label({agc_labels.shape}), pred({agc_preds.shape}), start to draw joint plot"
    )

    print("R2", r2_score(agc_labels, agc_preds))
    print("pearsonr", pearsonr(agc_labels, agc_preds)[0])

    idxes = np.where(agc_labels > 0)[0]
    agc_labels = agc_labels[idxes]
    agc_preds = agc_preds[idxes]

    idxes = np.where(agc_labels < 10)[0]
    agc_labels = agc_labels[idxes]
    agc_preds = agc_preds[idxes]

    data = {"agc_labels": agc_labels, "agc_preds": agc_preds}

    data = pd.DataFrame(data)

    mpl.rcParams["font.size"] = 14

    fig = plt.figure(figsize=(9, 7))

    subfig, ax = plt.subplots(figsize=(8, 6))

    sns.lineplot(
        x=agc_labels,
        y=agc_preds,
    )
    sns.jointplot(x=agc_labels, y=agc_preds, kind="hex", color="#c08292")
    sns.despine(offset=10, trim=True)

    plt.xlabel(
        "Actual number of atoms", labelpad=10, fontdict={"size": 16, "family": "Arial"}
    )
    plt.ylabel(
        "Predicted number of atoms",
        labelpad=10,
        fontdict={"size": 16, "family": "Arial"},
    )

    # plt.savefig("./eps/agc.boxplot.eps", format="eps", bbox_inches="tight")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "agc.jointplot.png", dpi=300, format="png", bbox_inches="tight")

    end(f"draw end, file saved in {output_dir / 'agc.jointplot.png'}")


# Rotate the starting point around the cubehelix hue circle
@cli.command()
@click.option("--data", "-D", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def kde(
    data: Path,  # [N, N]
    output_dir: Path,
):
    emin=0
    emax=10
    lowdark=True
    bw_adjust=0.43
    figsize=(6, 6)

    file_name = data.stem

    start('load data')
    data = np.load(data)
    end('load data')

    # Set up the matplotlib figure
    # f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
    fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot()
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    # width_px, height_px = (
    #     fig.get_size_inches() * fig.dpi
    # )  # 获取宽度和高度（以像素为单位）
    # print(f"Width: {width_px}, Height: {height_px}")

    x = []
    y = []

    data = np.clip(data, emin, emax)

    # 归一化处理
    data = (data - emin) / (emax - emin)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i][j]
            if lowdark:
                num = 10 - int(value * 10)
            else:
                num = int(value * 2)
            for _ in range(num):
                x.append(i / 15 - 1)
                y.append(j / 15 - 1)

    # # Create a cubehelix colormap to use with kdeplot
    # cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    # x, y = rs.normal(size=(2, 50))
    ax = sns.kdeplot(
        x=x,
        y=y,
        # cmap=sns.cubehelix_palette(start=0.9, light=2, as_cmap=True),
        # cmap='twilight',
        cmap="PuBu",
        # color='#7EB5D6',
        fill=True,
        clip=(-5, 5),
        bw_adjust=bw_adjust,
        # cut=10,
        thresh=0,
        levels=15,
        # ax=ax,
        cbar=True,
    )
    ax.set_axis_off()

    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{file_name}.kde.png", dpi=300, format="png", bbox_inches="tight")

    end(f"draw end, file saved in {output_dir / f'{file_name}.kde.png'}")

    # plt.show()


# show(
#     np.sum(agc_data, axis=2, keepdims=False),
#     lowdark=False,
#     filename=f"{cifid}.zaxis",
#     emax=10,
#     figsize=(a, b),
# )
# show(
#     np.sum(agc_data, axis=0, keepdims=False),
#     lowdark=False,
#     filename=f"{cifid}.xaxis",
#     emax=25,
#     bw_adjust=0.43,
#     figsize=(b * math.sqrt(3) / 2, c),
# )


@cli.command()
@click.option("--agc-labels", "-L", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def distribution(agc_labels: Path, output_dir: Path):
    start("Load agc labels")

    agc_labels = np.load(agc_labels)

    end("Load end, start to draw")

    data = {"agc_labels": agc_labels}
    data = pd.DataFrame(data)

    ax = sns.histplot(
        x="agc_labels",
        data=data,
        stat="probability",
        binwidth=1,
        color="#c08292",
    )

    plt.xlabel("Number of atoms", labelpad=10, fontdict={"size": 16, "family": "Arial"})
    plt.ylabel("Ratio", labelpad=10, fontdict={"size": 16, "family": "Arial"})
    plt.title(
        "Distribution of number of atoms",
        pad=20,
        fontdict={"size": 20, "family": "Arial"},
    )

    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"agc.distribution.png", dpi=300, format="png", bbox_inches="tight")

    end(f"draw end, file saved in {output_dir / f'agc.distribution.png'}")

if __name__ == '__main__':
    cli()
