# MOFTransformer version 2.0.0
class CLICommand:
    """
    Make GRIDAY which calculated potential basis function.
    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        from spbnet.data.install_griday import make_griday

        make_griday()
