from unimol_tools import MolTrain
import numpy as np
from pathlib import Path
import yaml
import joblib
import math
import pandas as pd

with open("./config.yaml", "r") as f:
    config = yaml.full_load(f)

root_dir = Path(config["root_dir"])
unimol_dir = root_dir / "unimol"

splits = ["train", "val", "test"]
custom_datas = {
    split: joblib.load(str(unimol_dir / f"{split}.joblib")) for split in splits
}
dfs = {
    split: pd.read_csv(root_dir / f"{config['id_prop']}.{split}.csv")
    for split in splits
}
for split in splits:
    custom_datas[split]["target"] = dfs[split][config["target"]]


def merge(data1, data2):
    atoms = data1["atoms"] + data2["atoms"]
    coordinates = data1["coordinates"] + data2["coordinates"]
    l1 = len(data1["atoms"])
    l2 = len(data2["atoms"])
    kfold = round((l1 + l2) / l2)
    if kfold - ((l1 + l2) / l2) > 0.3:
        kfold = math.floor((l1 + l2) / l2)  # more training data
    print(f"KFOLD: {kfold}, origin: {(l1 + l2) / l2}")
    target = np.concatenate([data1["target"], data2["target"]])

    return {"atoms": atoms, "coordinates": coordinates, "target": target}, kfold


custom_datas["merge"], kfold = merge(custom_datas["train"], custom_datas["val"])

clf = MolTrain(
    task="regression",
    metrics="mae",
    use_cuda=True,
    save_path=config["log_dir"],
    batch_size=config["batch_size"],
    epochs=config["epochs"],
    early_stopping=config["early_stopping"],
    kfold=kfold,
    target_anomaly_check=None,
)
clf.fit(custom_datas["merge"])
