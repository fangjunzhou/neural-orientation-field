import pathlib
from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """Neural Orientation Field project config dataclass.

    Attributes: 
        input_path: the path containing all the input images.
        output_path: the path to store the trained model.
        cache_path: the path for cache including COLMAP database, etc.
    """
    input_path: pathlib.Path
    output_path: pathlib.Path
    cache_path: pathlib.Path

    def to_dict(self):
        return {
            "input_path": str(self.input_path.resolve()),
            "output_path": str(self.output_path.resolve()),
            "cache_path": str(self.cache_path.resolve()),
        }

    @staticmethod
    def from_dict(config_dict: dict):
        return ProjectConfig(
            input_path=pathlib.Path(config_dict["input_path"]),
            output_path=pathlib.Path(config_dict["output_path"]),
            cache_path=pathlib.Path(config_dict["cache_path"])
        )


def confirm():
    """Ask for confirm.

    Returns:
        True if confirmed.
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("OK to push to continue [Y/N]? ").lower()
    return answer == "y"
