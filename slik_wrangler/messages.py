"""
Creates functionality that colors log messages
"""

from colorama import Fore


DANGER = Fore.RED
WARNING = Fore.YELLOW
SUCCESS = Fore.GREEN
NORMAL = Fore.RESET
INFO = Fore.CYAN


def log(*messages, code='normal', sep=' ', end='\n', file=None):
    """
    Distinguishes log messages from print statements.
    Works like a normal print statement but inclusive of colors
    :param messages: Message to be logged
    :param code: Log significance
    :return: distinguished log message
    """

    validated_codes = {
        'danger': DANGER,
        'warning': WARNING,
        'success': SUCCESS,
        'normal': NORMAL,
        'info': INFO
    }

    validated_codes_keys = validated_codes.keys()

    assert code in validated_codes_keys, \
        f"Message code most be one of {list(validated_codes_keys)}"

    message = sep.join(messages)

    print(validated_codes[code] + message, end=end, file=file)
