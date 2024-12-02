from dataclasses import dataclass


def confirm():
    """Ask for confirm.

    Returns:
        True if confirmed.
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("OK to push to continue [Y/N]? ").lower()
    return answer == "y"
