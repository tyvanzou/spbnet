from ase.io import read
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import random
from spbnet.utils.echo import title, start, end, param


def calcAtomNum(root_dir: str, num: int = 500):
    title("CALC ATOM NUM")

    start("Start to calculate atom num")
    root_dir = Path(root_dir)
    param(root_dir=root_dir.absolute(), sample_num=num)

    cif_file_paths = list(root_dir.iterdir())
    random.shuffle(cif_file_paths)
    cif_file_paths = cif_file_paths[:num]
    atom_nums = []
    for cif_file_path in tqdm(cif_file_paths):
        try:
            atoms = read(cif_file_path.absolute())
            atom_nums.append(len(atoms.get_atomic_numbers()))
        except:
            continue
    atom_nums = np.array(atom_nums)

    mean, std = atom_nums.mean(), atom_nums.std()
    end(
        f'Calc end, mean={mean}, std={std}. Suggest to use {mean} or {mean+std} as the "max_graph_len" hyper parameter.'
    )
    title("CALC ATOM NUM END")
