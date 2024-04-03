import pandas as pd
from pathlib import Path
from spbnet.utils.echo import title, param, start, end


def calcMeanStd(file_path: str, task: str, sample_num: int = 500):
    title("CALC MEAN STD")
    start("Star to calculate mean and std")
    file_path = Path(file_path)
    param(file=file_path.absolute(), task=task, sample_num=sample_num)
    df = pd.read_csv(file_path.absolute())
    df = df.dropna(subset=[task])
    df = df.sample(n=sample_num)
    mean = df[task].mean()
    std = df[task].std()
    end(f"Calculate end, mean={mean}, std={std}")
    title("CALC MEAN STD END")
