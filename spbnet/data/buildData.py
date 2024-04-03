from .prepare_data import make_prepared_data
from .float2int8 import float2int8
from pathlib import Path
from multiprocessing import Process
import shutil
import argparse
from tqdm import tqdm
from spbnet.utils.echo import title, err, param, start, end


def buildData(
    root_dir, cif_dir, target_dir, n_process, crystal_max_length, max_num_unique_atoms
):
    def buildGraphAndGrid(cif_path: Path, target_dir: Path):
        cifid = cif_path.stem

        def ifProcessed():
            suffix_map = {
                "graphdata": "graphdata",
                "grid": "grid",
                "griddata": "griddata",
            }
            for suffix in ["graphdata", "grid", "griddata"]:
                if not (
                    target_dir / suffix / f"{cifid}.{suffix_map.get(suffix)}"
                ).exists():
                    return False
            return True

        if ifProcessed():
            return

        total_dir = target_dir / "total"
        total_dir.mkdir(exist_ok=True)
        make_prepared_data(
            cif_path,
            total_dir,
            max_length=crystal_max_length,
            max_num_unique_atoms=max_num_unique_atoms,
        )
        # move to target folder
        for suffix in ["graphdata", "grid", "griddata"]:
            (target_dir / suffix).mkdir(exist_ok=True)
            if not (total_dir / f"{cifid}.{suffix}").exists():
                continue
            shutil.move(
                (total_dir / f"{cifid}.{suffix}"),
                (target_dir / suffix / f"{cifid}.{suffix}"),
            )

    def process_cif(cifpath: Path, target_dir: Path):
        buildGraphAndGrid(cifpath, target_dir)

    def process_multi_cif(cifpaths, target_dir: Path):
        for cifpath in tqdm(cifpaths):
            try:
                process_cif(cifpath, target_dir)
            except Exception as e:
                err(f"Error when processing {cifpath.stem}: {e}")

    root_dir = Path(root_dir)
    cif_dir = root_dir / cif_dir
    target_dir = root_dir / target_dir
    target_dir.mkdir(exist_ok=True)

    cif_paths = list(cif_dir.iterdir())
    cif_paths = list(filter(lambda item: item.suffix == ".cif", cif_paths))

    title("DATA PREPROCESSING")
    param(
        cif_dir=cif_dir.absolute(),
        target_dir=target_dir.absolute(),
        cif_num=len(cif_paths),
    )

    start("Process [1/2]: prepare modal data")

    process_num = n_process

    cif_num_per_process = len(cif_paths) // process_num + 1
    processes = []
    for i in range(process_num):
        cif_paths_i = cif_paths[i * cif_num_per_process : (i + 1) * cif_num_per_process]
        process = Process(target=process_multi_cif, args=(cif_paths_i, target_dir))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    end("Process [1/2] end")

    start("Process [2/2]: float2int8")

    float2int8(target_dir, n_process=process_num)

    end("Process [2/2] end")

    title("DATA PREPROCESSING END")
