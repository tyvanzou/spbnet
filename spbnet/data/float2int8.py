import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def handle_griddata(file_griddata, emin=-5000.0, emax=5000, bins=101):
    griddata = np.fromfile(file_griddata.absolute(), dtype=np.float32)
    griddata = griddata.reshape(30 * 30 * 30, 20)

    griddata[griddata <= emin] = emin
    griddata[griddata > emax] = emax

    x = np.linspace(emin, emax, bins)
    griddata = np.digitize(griddata, x) + 1

    griddata = griddata.astype(np.uint8)

    return griddata


def process(data_path: Path, target_dir: Path):
    arr = handle_griddata(data_path)
    np.save((target_dir / f"{data_path.stem}").absolute(), arr)


def float2int8(dataset_dir: Path, n_process: int):
    griddata_dir = dataset_dir / "griddata"
    target_dir = dataset_dir / "griddata8"
    target_dir.mkdir(exist_ok=True)

    data_paths = list(griddata_dir.iterdir())

    with Pool(n_process) as pool:
        result = list(
            tqdm(
                pool.imap_unordered(
                    partial(process, target_dir=target_dir), data_paths
                ),
                total=len(data_paths),
            )
        )
