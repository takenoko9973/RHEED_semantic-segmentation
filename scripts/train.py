from pathlib import Path

from rheed_segmentation.config.experiment_config import Configs
from rheed_segmentation.experiment import training_experiments

common_config_file = Path("configs", "common.yaml")

config_files = [
    Path("configs", "raw.yaml"),
    # Path("configs", "CLAHE.yaml"),
    # Path("configs", "CLAHE_Gaussian.yaml"),
]


def main(config_files: list[Path]) -> None:
    configs = Configs.model_validate(
        {
            "config_paths": config_files,
            "common_config_path": common_config_file,
        }
    )
    training_experiments(configs)


if __name__ == "__main__":
    main(config_files)
