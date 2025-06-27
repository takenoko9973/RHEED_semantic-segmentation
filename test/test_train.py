from pathlib import Path

from rheed_segmentation.config import Configs
from rheed_segmentation.experiment import training_experiments

config_path = Path(__file__).parent / "configs" / "test.yaml"
common_config_path = Path(__file__).parent / "configs" / "common.yaml"


def main() -> None:
    config = Configs.model_validate(
        {"config_paths": [config_path], "common_config_path": common_config_path}
    )
    training_experiments(config)


if __name__ == "__main__":
    main()
