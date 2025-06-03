import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / "../src").resolve()))

from rheed_segmentation.config import load_configs
from rheed_segmentation.experiment import experiments

config_file = Path(__file__).parent / "test_config.yaml"


def main() -> None:
    config = load_configs(config_file)
    experiments(config)


if __name__ == "__main__":
    main()
