from spbnet.data.checkData import checkData
import argparse


class CLICommand:
    """
    This command is used to check if the modal data (build by command `spbnt build-data`) is correct.

    If the data are properly built. The directory should look like:

    - root_dir
        - cif
            xxx.cif
        - spbnet
            - graphdata
                xxx.graphdata
            - grid
                xxx.grid
            - griddata8
                xxx.npy
        benchmark.train.csv
        benchmark.validate.csv
        benchmark.test.csv

    This command will check data in this form.
    Use it like:

    $ spbnet check_data --root-dir PATH/TO/ROOT/DIR --modal-dir spbnet --prop-dir . --prop-name benchmark --split all

    --root-dir is required, others are optional. Note prop_dir is relative to root_dir.
    By default, this command will check [train, validate, test].
    You can change the behavior like "--split none" to check "benchmark.csv" or "--split other" to check "benchmark.other.csv"
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--root-dir", type=str, required=True, help="")
        parser.add_argument("--modal-dir", type=str, default="spbnet")
        parser.add_argument("--prop-dir", type=str, default=".")
        parser.add_argument("--prop-name", type=str, default="benchmark")
        parser.add_argument("--split", type=str, default="all")

    @staticmethod
    def run(args):
        if args.split == "none":
            args.split = None
        checkData(
            args.root_dir, args.modal_dir, args.prop_dir, args.prop_name, args.split
        )
