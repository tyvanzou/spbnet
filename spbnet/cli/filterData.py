from spbnet.data.filterData import filterData
import argparse


class CLICommand:
    """
    This command is to filter data which are preprocess uncorrectly.

    The directory should look like:

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

    This command will filter data in this form.
    Use it like:

    $ spbnet filter-data --root-dir PATH/TO/YOUR/DIR --modal-dir spbnet --prop-dir . --prop-name benchmark --split all

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
        parser.add_argument(
            "--outlier",
            type=float,
            help="By default, SpbNet will filter outlier data point. outlier is defined by 'Q1 - outlier * IQR' and 'Q3 + outlier * IQR'. Set outlier < 0 to cancel the preserve all the data",
            default=5,
        )

    @staticmethod
    def run(args):
        filterData(
            args.root_dir,
            args.modal_dir,
            args.prop_dir,
            args.prop_name,
            outlier=args.outlier,
        )
