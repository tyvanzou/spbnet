import pandas as pd
from pathlib import Path

from ..utils.echo import title, param, start, end


def calcMeanStd(file_path: str):
    """
    coprresponding file should like
    cifid, task1, task2..
    CIF1, 1.233, 1.293...
    CIF2, 2.313, 1.393...
    """
    title("CALC MEAN STD")
    start("Star to calculate mean and std")
    param(file=file_path.absolute(), task=task)
    df = pd.read_csv(file_path.absolute())
    tasks = df.columns[1:]
    tasks = [str(task) for task in tasks]
    end(f"Calculate end")
    for task in tasks:
        dropna_df = df.dropna(subset=[task])
        print(
            f"{task}: mean={float(dropna_df[task].mean())}, std={float(dropna_df[task].std())}"
        )
    title("CALC MEAN STD END")


@click.command()
@click.option(
    "--id-prop", "-I", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def calcMeanStd(id_prop: Path):
    calcMeanStd(id_prop)
