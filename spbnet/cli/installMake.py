# MOFTransformer version 2.0.0
class CLICommand:
    """
    Install make which is needed to build GRIDAY(to generate potential basis function).
    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        from spbnet.data.install_griday import install_make

        install_make()
