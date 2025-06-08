from pathlib import Path

from rheed_segmentation.config.experiment_config import load_configs
from rheed_segmentation.experiment import training_experiments

common_config_file = Path("configs", "common.yaml")

config_files = [
    Path("configs", "raw.yaml"),
    Path("configs", "CLAHE.yaml"),
    Path("configs", "CLAHE_Gaussian.yaml"),
]


def main(config_files: list[Path]) -> None:
    config = load_configs(config_files, common_config_file)
    training_experiments(config)


if __name__ == "__main__":
    main(config_files)
