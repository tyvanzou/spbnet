from unimol_tools import MolPredict
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

test_data = joblib.load(str(unimol_dir / f"test.joblib")) 
df = pd.read_csv(root_dir / f"{config['id_prop']}.test.csv")
test_data["target"] = df[config["target"]]

dataset = Path(config["root_dir"]).stem
clf = MolPredict(load_model=config['log_dir'])
y_pred = clf.predict(data = test_data, save_path=f'./test_result/{dataset}')

test_result_df = pd.DataFrame()
test_result_df['target'] = df[config["target"]]
test_result_df['predict'] = y_pred
test_result_df.to_csv(f'./test_result/{dataset}/test_result.csv', index=False)

# clf = MolTrain(
#     task="regression",
#     metrics="mae",
#     use_cuda=True,
#     save_path=config["log_dir"],
#     batch_size=config["batch_size"],
#     epochs=config["epochs"],
#     early_stopping=config["early_stopping"],
#     kfold=kfold,
#     target_anomaly_check=None,
# )
# clf.fit(custom_datas["merge"])
