# from moftransformer.examples import example_path
from moftransformer.utils import prepare_data
import pandas as pd
import numpy as np
import json
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import pickle
import torch
import os

import yaml
import click

# dataset = 'hmof'
# downstreams = [
#   'CO2-298-2.5',
#   # "CO2-298-0.1",
#   'CH4-298-2.5',
#   # "CH4-298-35",
#   'N2-298-0.9',
#   'H2-77-2',
#   'Kr-273-10',
#   'Xe-273-10',
# ]

# dataset = 'hmofheat'
# downstreams = ['CO2-298K-2.5bar', 'H2-77K-2bar', 'CH4-298K-2.5bar', 'N2-298K-0.9bar']

# dataset = 'coremof'
# downstreams = ["N2-77-100000", "Ar-87-1"]

# dataset = 'cof'
# downstreams = ['highbar', 'logkh', 'lowbar', 'qst']

# dataset = 'zeolite'
# downstreams = ['heat_of_adsorption', 'unitless_KH']

# dataset = 'ppn'
# downstreams = ['1bar', '65bar']

# dataset = 'catalysis'
# downstreams = ['energy']

# dataset = 'aromatic'
# downstreams = ['Np','Sp']

# dataset = "ch4n2"
# # downstreams = ["ch4n2ratio-0.1bar", "ch4n2ratio-1bar", "ch4n2ratio-10bar"]
# downstreams = ['KCH4', 'KN2']

# dataset = "heat"
# downstreams = ['Cv_molar_250.00', 'Cv_gravimetric_250.00', 'Cv_molar_275.00', 'Cv_gravimetric_275.00']

# dataset = "krxe"
# downstreams = ["Qst-CH4-298k-6bar","Qst-H2-77k-6bar","Qst-H2-130k-6bar","Qst-H2-200k-6bar","Qst-H2-243k-6bar","Select-Xe-Kr-1bar","Select-Xe-Kr-5bar"]

# dataset = "krxen"
# downstreams = [
#     "K_Xe",
#     "K_Kr",
#     "Q_Xe",
#     "Q_Kr",
#     "P_Xe",
#     "P_Kr",
#     "Sperm_Xe_Kr",
#     "Sperm_Kr_Xe",
# ]

# dataset = 'mechanical'
# downstreams = ["KVRH", "GVRH"]

# dataset = 'amorphous'
# downstreams = ["kr-273k-1bar"]

# dataset = "n2o2"
# downstreams = [
#     "SelfdiffusionofO2cm2s",
#     "SelfdiffusionofN2cm2s",
#     "O2N2ratio",
#     "HenrysconstantO2",
#     "HenrysconstantN2",
#     "SelfdiffusionofO2cm2sinfDilute",
#     "SelfdiffusionofN2cm2sinfDilute",
#   ]

# dataset = 'tsd'
# downstreams = ['tsd']

# dataset = 'aromatic'
# downstreams = ['Np', 'Sp']

# dataset = 'amorphous'
# downstreams = ['gcmc_loading_cm3cm3']

# dataset = 'c3h6c3h8'
# downstreams = ['C3H8_C3H6_Selectivity_1Bar']

# dataset = 'bandgap'
# downstreams = ['bandgap']

# dataset = 'ftstable'
# downstreams = ['5p', '25p', '50p', '100p', '200p']

# dataset = 'prestable'
# # downstreams = ['diffusivity_log', 'raspa_100bar']
# downstreams = ['diffusivity_log_25', 'diffusivity_log_50', 'diffusivity_log_75', 'diffusivity_log_100', 'raspa_100bar_25', 'raspa_100bar_50', 'raspa_100bar_75', 'raspa_100bar_100']

# dataset = 'bandgap'
# downstreams = ['bandgap']

# dataset = 'c3h6c3h8coremof'
# downstreams = ['C3H8_loadings','C3H6_loadings','C3H8_C3H6_Selectivity_1bar','TSN_S_1Bar',
#          'C3H8_Henry_298K','C3H6_Henry_298K','C3H8_C3H6_Selectivity_infinite']

# dataset = 'hmofheat'
# downstreams = ['CO2-298K-2.5bar', 'H2-77K-2bar', 'CH4-298K-2.5bar', 'N2-298K-0.9bar']


def cif(root_dir):
    root_dir = Path(root_dir)
    root_cifs = root_dir / "cif"
    root_dataset = root_dir / "moftransformer"
    downstream = (
        "NOTUSED"  # not used, we do not use the defualt split method of moftransformer
    )

    train_fraction = 0.8
    test_fraction = 0.1

    prepare_data(
        root_cifs,
        root_dataset,
        downstream=downstream,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        max_num_unique_atoms=512,  # default is 300, but 300 will filter some MOFs in CoREMOF
        max_length=120.0,  # NOTE: default is 60, too small
    )


def downstream(root_dir, task, id_prop="benchmark"):
    print(f"build downstream data for {task}, dfname is {df}")

    data_dir = root_dir / "moftransformer"

    def df2json(df: pd.DataFrame, task: str):
        ret = dict()
        for item in tqdm(df.iloc, total=len(df)):
            if not np.isnan(item[task]):
                ret[item["cifid"]] = item[task]
        return ret

    splits = ["train", "val", "test"]
    dfs = {split: pd.read_csv(root_dir / f"{id_prop}.{split}.csv") for split in splits}

    for split in splits:
        with (data_dir / f"{split}_{task}.json").open("w") as f:
            json.dump(df2json(dfs[split], task), f)


def modal(root_dir, task, n_process):
    print(f"move modal data for {task}")

    data_dir = root_dir / "moftransformer"

    for split in ["train", "val", "test"]:
        if not (data_dir / split).exists():
            (data_dir / split).mkdir()

        with open((data_dir / f"{split}_{task}.json").absolute(), "r") as f:
            data: dict = json.load(f)
        keys = list(data.keys())

        suffixs = ["graphdata", "grid", "griddata16"]

        def process(key):
            for suffix in suffixs:
                shutil.copy2(
                    (data_dir / f"total/{key}.{suffix}").absolute(),
                    (data_dir / f"{split}/{key}.{suffix}").absolute(),
                )

        with Pool(n_process) as pool:
            result = list(tqdm(pool.imap_unordered(process, keys), total=len(keys)))


def check(root_dir, task):
    def make_grid_data(grid_data, emin=-5000.0, emax=5000, bins=101):
        """
        make grid_data within range (emin, emax) and
        make bins with logit function
        and digitize (0, bins)
        ****
            caution : 'zero' should be padding !!
            when you change bins, heads.MPP_heads should be changed
        ****
        """
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax

        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1

        return new_grid_data

    def check_cifid(cifid: str, split: str):
        nonlocal data_dir
        try:
            GRID = 30
            channel = 20

            total_dir = data_dir / split
            graphdata = pickle.load((total_dir / f"{cifid}.graphdata").open("rb"))
            # graphdata = pickle.load(open(f"./spbnet/graphdata/{cifid}.graphdata", "rb"))
            file_grid = total_dir / f"{cifid}.grid"
            file_griddata = total_dir / f"{cifid}.griddata16"

            # get grid
            with file_grid.open("r") as f:
                lines = f.readlines()
                a, b, c = [float(i) for i in lines[0].split()[1:]]
                angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
                cell = [int(i) for i in lines[2].split()[1:]]

            grid_data = pickle.load(file_griddata.open("rb"))
            grid_data = make_grid_data(grid_data)
            grid_data = torch.FloatTensor(grid_data)

            grid_data = grid_data.reshape([GRID, GRID, GRID])

            return True
        except:
            return False

    data_dir = root_dir / "moftransformer"

    for task in downstreams:
        for split in ["train", "val", "test"]:
            print(f"CHECKING: {task}-{split}")

            with open((data_dir / f"{split}_{task}.json").absolute(), "r") as f:
                data: dict = json.load(f)
            cifids = list(data.keys())
            for cifid in tqdm(cifids):
                if not check_cifid(cifid, split):
                    print(f"ERROR: {cifid}")
                    exit(0)
                    # for suffix in ["graphdata", "grid", "griddata16"]:
                    #     os.remove((self_dir / "total" / f"{cifid}.{suffix}").absolute())
            print("SUCCESS")


@click.group()
def cli():
    pass


@cli.command()
def build():
    config = yaml.load(open("./config.yaml", "r"))
    root_dir = Path(config["root_dir"])
    df = config["id_prop"]
    downstreams = config["downstreams"]
    n_process = 8

    cif(root_dir)
    for task in downstreams:
        downstream(root_dir, task, df)
        modal(root_dir, task, n_process)
        check(root_dir, task)


@cli.command()
def check():
    config = yaml.load(open("./config.yaml", "r"))
    root_dir = Path(config["root_dir"])
    downstreams = config["downstreams"]
    for task in downstreams:
        check(root_dir, task)


if __name__ == "__main__":
    cli()
