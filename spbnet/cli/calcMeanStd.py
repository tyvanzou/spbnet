from spbnet.data.calcMeanStd import calcMeanStd
import argparse


class CLICommand:
    """
    This command is used to estimate the mean and std of the dataset.

    There is no need for spbnet to use the command.
    SpbNet will automatically calculate the mean and variance from the training label data and normalize it.
    This command can be used to train baseline model PMTransformer or MOFTransforemr.

    To use this command, please prepare a csv file including label data.
    The csv file should look like:

    cifid   task1
    mof1    1.3
    mof2    2.1
    mof3    0.9
    ...     ...
    """

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--file", type=str, help="path to csv file", required=True)
        parser.add_argument("--task", type=str, help="task name", required=True)
        parser.add_argument("--num", type=int, help="num to sample data", default=500)

    def run(args):
        calcMeanStd(args.file, args.task, args.num)
