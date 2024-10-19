# import click
# from scipy.stats import pearsonr
# import numpy as np
# from sklearn.metrics import r2_score
# import pandas as pd


# @click.command()
# @click.option("--file", "-F", type=str, default="test_results.csv")
# @click.option("--label-column", "-L", type=str, default="target")
# @click.option("--predict-column", "-P", type=str, default="predict")
# def metric(file, label_column, predict_column):
#     df = pd.read_csv(file)
#     df = df.dropna()
#     label = df[label_column]
#     predict = df[predict_column]
#     r2 = r2_score(label, predict)
#     pear = pearsonr(label, predict)[0]
#     mae = np.mean(np.abs(label - predict))
#     mse = np.mean((label - predict) ** 2)
#     print("MAE", mae, "MSE", mse, "Pearson", pear, "R2", r2)


# if __name__ == "__main__":
#     metric()

import pandas as pd
import numpy  as np

df = pd.read_csv('./test_results.csv')
df['error'] = np.abs(df['target'] - df['predict'])
df.to_csv('./test_result.csv', index=False)