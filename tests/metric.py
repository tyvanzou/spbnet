import pandas as pd
from sklearn.metrics import r2_score
import numpy as np


df = pd.read_csv("./data/benchmark.test.csv")
pred_df = pd.read_csv("./predict/version_2/test_result.csv")

target = df["CO2-298-2.5"]
pred = pred_df["predict"]

print(r2_score(target, pred), (target - pred).abs().mean())
