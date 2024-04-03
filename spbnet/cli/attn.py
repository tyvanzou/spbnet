import argparse
from spbnet.visualize.attn import attn
from pathlib import Path


class CLICommand:
    """
    Get attention score (html) according to a given weight.
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--cif-path",
            type=str,
            help="cif-path to get attention score",
            required=True,
        )
        parser.add_argument(
            "--modal-dir",
            type=str,
            help="directory contains modal data: graphdata, griddata8, grid",
            required=True,
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            help="checkpoint(weights)",
            required=True,
        )
        parser.add_argument(
            "--out-dir",
            type=str,
            help="output directory, default is current directory",
            default=".",
        )

    @staticmethod
    def run(args):
        attn(
            Path(args.cif_path),
            Path(args.modal_dir),
            Path(args.ckpt),
            Path(args.out_dir),
        )
