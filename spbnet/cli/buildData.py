from spbnet.data.buildData import buildData
import argparse


class CLICommand:
    """
    Build datas of different modals, since spbnet is a multi-modal network.

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
            - graphdata
                - mof1.graphdata
                - mof2.graphdata
                ...
            - grid
                - mof1.grid
                - mof2.grid
                ...
            - griddata8
                - mof1.npy
                - mof2.npy
                ...
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--root-dir", type=str, help="root_dir to build modal data", required=True
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
        parser.add_argument(
            "--crystal-max-length",
            type=float,
            help="In order to ensure that the energy grid resolution is greater than 3A, a supercell needs to be constructed. crystal_max_length is the maximum lattice constant length of the supercell, the default is 120.0. See papers of SpbNet and MOFTransformer for more information",
            default=120.0,
        )
        parser.add_argument(
            "--max-num-unique-atoms",
            help="A crystal with too many atoms in a single unit cell will lead to large errors, so the number of atoms needs to be limited. The default limit is 512. For tobacco dataset we suggest to use 1024. See MOFTransformer paper for more information",
            type=int,
            default=512,
        )

    @staticmethod
    def run(args):
        buildData(
            args.root_dir,
            args.cif_dir,
            args.target_dir,
            args.n_process,
            args.crystal_max_length,
            args.max_num_unique_atoms,
        )
