from pathlib import Path

from rheed_segmentation.config import load_config
from rheed_segmentation.experiment import training_experiment

config_file = Path(__file__).parent / "test_config.yaml"


def main() -> None:
    config = load_config(config_file)
    training_experiment(config)


if __name__ == "__main__":
    main()
