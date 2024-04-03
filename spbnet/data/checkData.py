import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from spbnet.utils.echo import start, end, title, err, param


def checkData(root_dir, modal_dir, prop_dir, prop_name, split):

    root_dir = Path(root_dir)
    modal_dir = root_dir / modal_dir
    prop_dir = root_dir / prop_dir

    def check_cifid(cifid: str):
        try:
            GRID = 30
            channel = 20

            data_dir = modal_dir
            graphdata = pickle.load(
                (modal_dir / "graphdata" / f"{cifid}.graphdata").open("rb")
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

            return None
        except Exception as e:
            return e

    title("CHECKING MODAL DATA")
    param(modal_dir=modal_dir.absolute(), prop_name=prop_name)

    def check_idprop(idprop_path):
        start(f"Checking {idprop_path.absolute()}")
        df = pd.read_csv(idprop_path.absolute())

        for item in tqdm(df.iloc, total=len(df)):
            cifid = item["cifid"]
            res = check_cifid(cifid)
            if res is not None:
                err(f"Error when checing {cifid}: {res}")
                return
        end(f"Check {idprop_path.absolute()} success")

    if split == "all":
        for split in ["train", "test", "validate"]:
            idprop_path = prop_dir / f"{prop_name}.{split}.csv"
            check_idprop(idprop_path=idprop_path)
    elif split is None:
        idprop_path = prop_dir / f"{prop_name}.csv"
        check_idprop(idprop_path)
    else:
        idprop_path = prop_dir / f"{prop_name}.{split}.csv"
        check_idprop(idprop_path)

    title("CHECK MODAL DATA END")
