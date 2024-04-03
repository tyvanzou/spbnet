import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import r2_score
from einops import rearrange
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
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
    dis = np.sqrt(ss)

    # if ss == 25 or ss == 50 or ss == 49:
    #     return dis
    # else:
    #     return 0

    return dis


show = np.load(f"./feat/{dataset}/self_attns.npy")
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
plt.savefig(
    "./png/self_attn.finegrained.png", dpi=300, format="png", bbox_inches="tight"
)
plt.savefig(
    "./eps/self_attn.finegrained.eps", dpi=600, format="eps", bbox_inches="tight"
)
# plt.savefig("./self_attn_center.png", dpi=300, format="png", bbox_inches="tight")
# plt.savefig("./self_attn.png", dpi=300, format="png", bbox_inches="tight")
