from unimol_tools import MolTrain
import numpy as np
from pathlib import Path
import yaml
import joblib
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

clf = MolTrain(
    task="regression", metrics="mae", use_cuda=True, save_path=config["log_dir"], batch_size=config['batch_size'], epochs=config['epochs'], early_stopping=config['early_stopping'], kfold=config['kfold']
)
clf.fit(custom_datas["train"])
