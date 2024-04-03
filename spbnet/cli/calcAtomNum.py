from spbnet.data.calcAtomNum import calcAtomNum
import argparse


class CLICommand:
    """
    This command is used to estimate the average number of atoms in a dataset.

    This is one of the key configuration for SpbNet.
    SpbNet is based on Transformer architecture.
    The sequence need to be truncated to keep smaller than the specified max length of tokens.
    We suggest setting the "max_graph_len" hyper parameter of spbnet similar to the average atom number to maintain relatively complete information of structure.
    Low "max_graph_len" hyper parameter will result in low accuracy.

    To calculate the average atom nunber, please use:

    $ spbnet calc_atom_num PATH/TO/YOUR/CIF_DIR

    The PATH/TO/YOUR/CIF_DIR should look like:

    - root_dir
        mof1.cif
        mof2.cif
        ...
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--root-dir", type=str, help="root_dir containing cif files"
        )
        parser.add_argument(
            "--num", type=int, default=500, help="num of sample files, default is 500"
        )

    def run(args):
        calcAtomNum(args.root_dir, args.num)
