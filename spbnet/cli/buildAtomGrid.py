import argparse
from spbnet.data.buildAtomGrid import buildAtomGrid


class CLICommand:
    """
    Build Atom Grid data for pretrain spbnet

    Please provide directory like:

    - root_dir
        - cif
            - mof1.cif
            - mof2.cif
            ...

    This command will produce:

    - root_dir
        - cif
            - mof1.cif
            - mof2.cif
            ...
        - spbnet
            - atomgrid
                - mof1.npy
                - mof2.npy
                ...
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--root-dir", type=str, help="root_dir to build atom grid", required=True
        )
        parser.add_argument(
            "--cif-dir",
            type=str,
            help='cif_dir to use, default is "cif", this command will use cif files from root_dir/cif_dir',
            default="cif",
        )
        parser.add_argument(
            "--target-dir",
            type=str,
            help="target directory to build data, this command will produce data in root_dir/target_dir",
            default="spbnet",
        )
        parser.add_argument(
            "--n-process",
            type=int,
            help="multiprocessing, number of processes to use, default is 1",
            default=1,
        )

    @staticmethod
    def run(args):
        buildAtomGrid(args.root_dir, args.cif_dir, args.target_dir, args.n_process)
