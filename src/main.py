from pathlib import Path

from rheed_segmentation.config.experiment_config import load_config
from rheed_segmentation.experiment import experiments

config_files = [
    Path("configs", "raw.yaml"),
    Path("configs", "CLAHE.yaml"),
    Path("configs", "CLAHE_Gaussian.yaml"),
]


def main(config_file: Path) -> None:
    config = load_config(config_file)
    experiments(config)


if __name__ == "__main__":
    main(config_files)
