import click
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


def metric(file, label_column, predict_column):
    df = pd.read_csv(file)
    df = df.dropna()
    label = df[label_column]
    predict = df[predict_column]
    r2 = r2_score(label, predict)
    pear = pearsonr(label, predict)[0]
    mae = np.mean(np.abs(label - predict))
    mse = np.mean((label - predict) ** 2)
    print("MAE", mae, "MSE", mse, "Pearson", pear, "R2", r2)


tasks = ['hmof', 'coremof', 'ch4n2', 'qmof', 'tsd']
for task in tasks:
    print(task)
    if task == 'cof/lowbar':
        version = 1
    else:
        version = 0
    metric(f'./{task}.csv', 'target', 'predict')
