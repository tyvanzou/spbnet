from pathlib import Path
from multiprocessing import Process
import shutil
import argparse
import click
from tqdm import tqdm

from .prepare_data import make_prepared_data
from .float2int8 import float2int8
from ..utils.echo import title, err, param, start, end


def buildModal(
    root_dir: Path,
    cif_folder: str,
    modal_folder: str,
    n_process: int,
    crystal_max_length: int,
    max_num_unique_atoms: int,
):
    def buildGraphAndGrid(cif_path: Path, modal_dir: Path):
        cifid = cif_path.stem

        def ifProcessed():
            suffix_map = {
                "graphdata": "graphdata",
                "grid": "grid",
                "griddata": "griddata",
            }
            for suffix in ["graphdata", "grid", "griddata"]:
                if not (
                    modal_dir / suffix / f"{cifid}.{suffix_map.get(suffix)}"
                ).exists():
                    return False
            return True

        if ifProcessed():
            return

        total_dir = modal_dir / "total"
        total_dir.mkdir(exist_ok=True)
        make_prepared_data(
            cif_path,
            total_dir,
            max_length=crystal_max_length,
            max_num_unique_atoms=max_num_unique_atoms,
        )
        # move to modal folder
        for suffix in ["graphdata", "grid", "griddata"]:
            (modal_dir / suffix).mkdir(exist_ok=True)
            if not (total_dir / f"{cifid}.{suffix}").exists():
                continue
            shutil.move(
                (total_dir / f"{cifid}.{suffix}"),
                (modal_dir / suffix / f"{cifid}.{suffix}"),
            )

    def process_cif(cifpath: Path, modal_dir: Path):
        buildGraphAndGrid(cifpath, modal_dir)

    def process_multi_cif(cifpaths, modal_dir: Path):
        for cifpath in tqdm(cifpaths):
            try:
                process_cif(cifpath, modal_dir)
            except Exception as e:
                err(f"Error when processing {cifpath.stem}: {e}")

    cif_dir = root_dir / cif_dir
    modal_dir = root_dir / modal_dir
    modal_dir.mkdir(exist_ok=True)

    cif_paths = list(cif_dir.iterdir())
    cif_paths = list(filter(lambda item: item.suffix == ".cif", cif_paths))

    title("DATA PREPROCESSING")
    param(
        cif_dir=cif_dir.absolute(),
        modal_dir=modal_dir.absolute(),
        cif_num=len(cif_paths),
    )

    start("Process [1/2]: prepare modal data")

    process_num = n_process

    cif_num_per_process = len(cif_paths) // process_num + 1
    processes = []
    for i in range(process_num):
        cif_paths_i = cif_paths[i * cif_num_per_process : (i + 1) * cif_num_per_process]
        process = Process(target=process_multi_cif, args=(cif_paths_i, modal_dir))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    end("Process [1/2] end")

    start("Process [2/2]: float2int8")

    float2int8(modal_dir, n_process=process_num)

    end("Process [2/2] end")

    title("DATA PREPROCESSING END")


@click.command()
@click.option(
    "--root-dir", "-R", type=click.Path(exists=True, file_okay=False, type=Path)
)
@click.option("--cif-folder", type=str, default="cif")
@click.option("--modal-folder", type=str, default="spbnet")
@click.option("--n-process", type=int, default=1)
@click.option("--crystal-max-length", type=float, default=120)
@click.option("--max-num-unique-atoms", type=float, default=512)
def buildModalCli(
    root_dir,
    cif_folder,
    modal_folder,
    n_process,
    crystal_max_length,
    max_num_unique_atoms,
):
    buildModal(
        root_dir,
        cif_folder,
        modal_folder,
        n_process,
        crystal_max_length,
        max_num_unique_atoms,
    )


if __name__ == "__main__":
    buildModalCli()
