import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import r2_score
from einops import rearrange
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()
dataset = args.dataset


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
    # dis = np.sqrt(ss)

    # if ss == 25 or ss == 50 or ss == 49:
    #     return dis
    # else:
    #     return 0

    return ss


@click.command()
@click.option('--self-attns', '-S', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--out-dir', '-O', type=click.Path(file_okay=False, path_type=Path))
def main(self_attns: Path, out_dir: Path):
    start("Start to load self attns")
    show = np.load(self_attns)
    end("Load end, attn shape: ", show.shape)

    show = rearrange(show.reshape((-1, 10, 10, 10, 1000)), "b z y x a -> b x y z a")


    # Scatter plot
    distance_matrix = np.array([[distance(i, j) for j in range(1000)] for i in range(1000)])
    # uniq_distance = np.unique(distance_matrix)
    # sort_distance = np.sort(uniq_distance)

    end("Distance matrix generate end")

    rand_indices = np.random.randint(0, show.shape[1], size=30)
    show = show[rand_indices]
    distance_matrix = np.tile(distance_matrix, [len(rand_indices), 1, 1])

    end("Sample end", show.shape, distance_matrix.shape)

    show = show.reshape(-1)
    distance_matrix = distance_matrix.reshape(-1)

    # for dis in sort_distance:
    #     mean_attn = np.mean(show[np.where(distance_matrix == dis)])
    #     print(f"Dis: {dis}, attn: {mean_attn}")

    # rand_indices = np.where(distance_matrix < 150)
    # show = show[rand_indices]
    # distance_matrix = distance_matrix[rand_indices]

    # rand_indices = np.where(show > 0.01)
    # show = show[rand_indices]
    # distance_matrix = distance_matrix[rand_indices]

    rand_indices = np.random.randint(0, show.shape[0], size=2000000)
    show = show[rand_indices]
    distance_matrix = distance_matrix[rand_indices]

    df = {"attention": show, "distance": distance_matrix}
    df = pd.DataFrame(df)

    # sns.scatterplot(x=distance_matrix, y=show, s=5)

    # plt.savefig("./scatter.png", dpi=300, format="png")
    # plt.savefig("./scatter.eps", dpi=600, format="eps")

    # sns.kdeplot(x=distance_matrix, y=show, cmap="Blues", fill=True)
    # plt.savefig("./kde.png", dpi=300, format="png")
    # plt.savefig("./kde.eps", dpi=600, format="eps")

    g = sns.lineplot(x=distance_matrix, y=show, errorbar="pi", color="#8389a0")
    g.set(ylim=[0, 0.06])
    g.set(xlim=[0, 100])
    sns.despine(offset=10, trim=True)
    # plt.xlim([0, 100])
    # plt.axis('auto')
    out_dir.mkdir(exists_ok=True, parents=True)
    plt.savefig(
        out_dir / "self_attn.lineplot.png", dpi=300, format="png", bbox_inches="tight"
    )

    end(f"File saved in {out_dir / 'self_attn.lineplot.png'}")

    # sns.lmplot(data=df, x="distance", y="attention")

if __name__ == '__main__':
    main()
