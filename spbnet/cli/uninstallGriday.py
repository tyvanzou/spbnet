# MOFTransformer version 2.0.0
class CLICommand:
    """
    Uninstall GRIDAY
    """

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        from spbnet.data.install_griday import uninstall_griday

        uninstall_griday()
