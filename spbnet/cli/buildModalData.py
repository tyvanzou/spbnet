from spbnet.data.buildData import buildData
import argparse
from spbnet.visualize.buildModalData import buildModalData
from pathlib import Path


class CLICommand:
    """
    build modal data for single cif
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--cif-path", type=str, help="cif-path to build modal data", required=True
        )
        parser.add_argument(
            "--target-dir", type=str, help="target-dir to build modal data", default="."
        )

    @staticmethod
    def run(args):
        buildModalData(Path(args.cif_path), Path(args.target_dir))
