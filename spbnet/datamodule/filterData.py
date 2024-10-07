from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import click
from traitlets import default

from ..utils.echo import title, start, end, param, warn


def filterData(root_dir: Path, modal_folder: str, id_prop: str, outlier: int):
    title("FILTER DATA")

    warn("We only check graphdata, grid, griddata8 shaped [30, 30, 30, 20]")

    root_dir = Path(root_dir)
    modal_dir = root_dir / modal_folder
    id_prop_path = root_dir / f"{id_prop}.csv"

    param(
        root_dir=root_dir.absolute(),
        modal_dir=modal_dir.absolute(),
        id_prop_path=id_prop_path.absolute(),
    )

    def check_cifid(cifid: str):
        try:
            GRID = 30
            channel = 20

            data_dir = modal_dir
            graphdata = pickle.load(
                (data_dir / "graphdata" / f"{cifid}.graphdata").open("rb")
            )
            # graphdata = pickle.load(open(f"./spbnet/graphdata/{cifid}.graphdata", "rb"))
            file_grid = data_dir / "grid" / f"{cifid}.grid"
            file_griddata = data_dir / "griddata8" / f"{cifid}.npy"

            # get grid
            with file_grid.open("r") as f:
                lines = f.readlines()
                a, b, c = [float(i) for i in lines[0].split()[1:]]
                angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
                cell = [int(i) for i in lines[2].split()[1:]]

            griddata = np.load(file_griddata.absolute())  # [30*30*30, 20] uint8
            griddata = griddata.astype(np.float32)

            griddata = griddata.reshape(GRID, GRID, GRID, channel)

            return True
        except:
            return False

    def get_value(df: pd.DataFrame, item, task: str):
        arr = df[task]
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        if outlier <= 0:
            return item[task], True

        low = q1 - outlier * iqr
        up = q3 + outlier * iqr

        if item[task] > up:
            warn(
                f"outlier point found when filtering data, cif: {item['cifid']}, task: {task}, value: {item[task]}. The Q1 (First Quartile) of this task is {q1}, Q3 (Third Quartile) is {q3}. To avoid significant errors, this data will be ignored."
            )
            # return np.percentile(df[task], 90)
            return np.nan, False
        if item[task] < low:
            warn(
                f"outlier point found when filtering data, cif: {item['cifid']}, task: {task}, value: {item[task]}. The Q1 (First Quartile) of this task is {q1}, Q3 (Third Quartile) is {q3}. To avoid significant errors, this data will be ignored for all models in this paper (including baseline model)."
            )
            # return np.percentile(df[task], 10)
            return np.nan, False
        return item[task], True

    start("Start to filter data")
    df = pd.read_csv((prop_dir / f"{prop_name}.csv").absolute())
    tasks = [str(column) for column in df.columns[1:]]
    param(tasks=tasks)
    cifids = df["cifid"]
    items = []
    outlier_count = 0
    for item in tqdm(df.iloc, total=len(df)):
        cifid = item["cifid"]
        for task in tasks:
            if not isinstance(item[task], float) and not isinstance(item[task], int):
                break
            if get_value(df, item, task)[1] is not True:
                outlier_count += 1
                break
        targets = [get_value(df, item, task)[0] for task in tasks]
        if not check_cifid(cifid):
            continue
        items.append([cifid] + targets)
    end(f"Filter end, All: {len(df)}, Filtered: {len(items)}, Outlier: {outlier_count}")
    start("Start to split train/validate/test dataset")
    # print(['cifid']+tasks, items[0])
    items = pd.DataFrame(items, columns=["cifid"] + tasks)
    items = items.sample(n=len(items))
    if len(items) > 7000:
        items = items.sample(7000)
    items = items.sample(n=len(items))
    train_ratio = 0.8
    val_ratio = 0.1
    size = len(items)
    items.to_csv((prop_dir / f"{prop_name}.filter.csv").absolute(), index=False)
    items[: int(size * train_ratio)].to_csv(
        (prop_dir / f"{prop_name}.train.csv").absolute(), index=False
    )
    items[int(size * train_ratio) : int(size * (train_ratio + val_ratio))].to_csv(
        (prop_dir / f"{prop_name}.val.csv").absolute(), index=False
    )
    items[int(size * (train_ratio + val_ratio)) :].to_csv(
        (prop_dir / f"{prop_name}.test.csv").absolute(), index=False
    )
    end("Split end")
    title("FILTER DATA END")


@click.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, type=Path)
)
@click.option("--modal-folder", "-M", type=str, default=".")
@click.option("--id-prop", "-I", type=str, default="benchmark")
@click.option("--outlier", "-O", type=int, default=3)
def filterDataCli(root_dir: Path, modal_folder: str, id_prop: str, outlier: int):
    filterData(root_dir, modal_folder, id_prop, outlier)


if __name__ == "__main__":
    filterDataCli()
