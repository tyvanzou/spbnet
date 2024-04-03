from colorama import Fore, Style
from pprint import pprint as pp

import os


# 获取终端宽度
def get_terminal_width():
    try:
      term_size = os.get_terminal_size().columns
    except:
      term_size = 50
    return term_size


terminal_width = get_terminal_width()


def title(sentence: str, len=terminal_width, charac="="):
    print(
        Fore.YELLOW
        + (" SpbNet: " + sentence + " ").center(len, charac)
        + Style.RESET_ALL
    )


def err(log: str):
    print(Fore.RED + 'ERROR: ' + log + Style.RESET_ALL)


def warn(log: str):
    print(Fore.YELLOW + "WARNING: " + log + Style.RESET_ALL)


def end(log: str):
    print(Fore.BLUE + log + Style.RESET_ALL)


def start(log: str):
    print(Fore.CYAN + log + Style.RESET_ALL)


def param(**kwargs):
    pp(kwargs)
