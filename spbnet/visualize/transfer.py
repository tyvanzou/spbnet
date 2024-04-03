import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Arial"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 12

fig, ax = plt.subplots()

datasets = [
    "pretrain",
    "coremof",
    "cof",
    "ppn",
    "zeolite",
]


def get_err(dataset: str):
    labels = np.load(f"./feat/{dataset}/agc_labels.npy").reshape(-1)
    preds = np.load(f"./feat/{dataset}/agc_preds.npy").reshape(-1)


    # idxes = np.where(labels > 0)[0]
    # labels = labels[idxes]
    # preds = preds[idxes]

    # idxes = np.where(labels < 10)[0]
    # labels = labels[idxes]
    # preds = preds[idxes]

    print(
        f"{dataset}: {r2_score(labels, preds)}, {pearsonr(labels, preds)[0]}, ratio: {np.sum(labels < 1) / len(labels)}"
    )

    rand_indices = np.random.randint(0, labels.shape[0], 500000)

    labels = labels[rand_indices]
    preds = preds[rand_indices]

    err = np.abs(labels - preds)
    return err


def get_distribution(dataset: str):
    labels = np.load(f"./feat/{dataset}/agc_labels.npy").reshape(-1)

    rand_indices = np.random.randint(0, labels.shape[0], 500000)

    labels = labels[rand_indices]

    return labels


df = {dataset.split("/")[0]: get_err(dataset) for dataset in datasets}
df = pd.DataFrame.from_dict(df)
df_melted = df.melt(
    id_vars=[],
    var_name="dataset",
    value_name="number",
    value_vars=["pretrain", "coremof", "cof", "ppn", "zeolite"],
)
df_melted["type"] = np.array(["error" for _ in range(len(df_melted))])


df = {dataset.split("/")[0]: get_distribution(dataset) for dataset in datasets}
df = pd.DataFrame.from_dict(df)
dis_df_melted = df.melt(
    id_vars=[],
    var_name="dataset",
    value_name="number",
    value_vars=["pretrain", "coremof", "cof", "ppn", "zeolite"],
)
dis_df_melted["type"] = np.array(["distribution" for _ in range(len(dis_df_melted))])

result_df = pd.concat([df_melted, dis_df_melted], axis=0)

# df = pd.DataFrame(df)
palette = ["#cfcdcd", "#97a0b2", "#c69892", "#57746b", "#9e6f7a"]

sns.violinplot(
    data=result_df,
    x="dataset",
    y="number",
    hue="type",
    ax=ax,
    # inner="box",
    inner="quart",
    fill=True,
    palette=palette,
    linewidth=1,
    split=True,
    density_norm="width",
    bw_adjust=3,
    bw_method="silverman",
)
# sns.pointplot(data=[np.mean(result_df)], markers='o', color='white', linestyles='-', scale=0.8)
# sns.despine(left=True, bottom=True)
plt.ylim(-2.5, 8)
plt.xlabel(
    "Dataset", labelpad=10, fontdict={"size": 18, "family": "Arial", "color": "black"}
)
plt.ylabel(
    "Number", labelpad=10, fontdict={"size": 18, "family": "Arial", "color": "black"}
)
plt.title(
    "Error distribution for different porous materials",
    pad=20,
    fontdict={"size": 18, "family": "Arial", "color": "black"},
)
# ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(
    labels=[dataset.split("/")[0] for dataset in datasets],
    rotation=45,
    ha="right",
    fontdict={"size": 18, "family": "Arial", "color": "black"},
)
ax.set_yticks([i for i in range(-2, 9, 2)])
ax.set_yticklabels(
    labels=[i for i in range(-2, 9, 2)],
    # rotation=45,
    # ha="right",
    fontdict={"size": 18, "family": "Arial", "color": "black"},
)
# plt.savefig("./test.png", dpi=300, bbox_inches='tight')
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.savefig("./png/transfer.png", format="png", dpi=300, bbox_inches="tight")
plt.savefig("./eps/transfer.eps", format="eps", dpi=300, bbox_inches="tight")
