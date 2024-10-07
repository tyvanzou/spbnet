from ase.io import read
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import random
import click

from ..utils.echo import title, start, end, param


def calcAtomNum(root_dir: Path, num: int = 500, vis=False):
    title("CALC ATOM NUM")

    start("Start to calculate atom num")
    param(root_dir=root_dir.absolute(), sample_num=num)

    cif_file_paths = list(root_dir.iterdir())
    random.shuffle(cif_file_paths)
    cif_file_paths = cif_file_paths[:num]
    atom_nums = []
    for cif_file_path in tqdm(cif_file_paths):
        try:
            atoms = read(cif_file_path.absolute())
            atom_nums.append(len(atoms.get_atomic_numbers()))
        except:
            continue
    atom_nums = np.array(atom_nums)

    if vis:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set(font_scale=1.3)
        # sns.set(font="Arial")

        # 使用Seaborn的histplot绘制柱状图
        sns.histplot(atom_nums)

        # 设置图表标题和坐标轴标签
        plt.title("Distribution of atom numbers")
        plt.xlabel("Number of atoms")
        plt.ylabel("Frequency")

        # 使用plt.tight_layout()自动调整子图参数
        plt.tight_layout()

        # 显示图表
        plt.savefig("./atom_number_distribution.png", dpi=300)

    mean, std = atom_nums.mean(), atom_nums.std()
    end(
        f'Calc end, mean={mean}, std={std}. Suggest to use {mean} or {mean+std} as the "max_graph_len" hyper parameter.'
    )
    title("CALC ATOM NUM END")


@click.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, type=Path)
)
@click.option("--num", type=int, default=500)
@click.option("--vis/--no-vis", default=False)
def calcAtomNumCli(root_dir, num, vis):
    calcAtomNum(root_dir, num, vis)


if __name__ == "__main__":
    calcAtomNumCli()
