from pathlib import Path

from rheed_segmentation.config.experiment_config import load_config
from rheed_segmentation.experiment import experiments

config_file = Path("config.yaml")


def main(config_file: Path) -> None:
    config = load_config(config_file)
    experiments(config)


if __name__ == "__main__":
    main(config_file)
