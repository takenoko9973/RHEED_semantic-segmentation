from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import albumentations as albu
import yaml

from rheed_segmentation.train import LossComputer

from .training_config import TrainingConfig
from .transform_config import TargetMode, TransformPipelineConfig


@dataclass
class ExperimentConfig:
    protocol: str
    data_dirs: list[Path]
    labels: dict[str, int]
    per_label: bool
    training: TrainingConfig
    transforms: TransformPipelineConfig
    comment: str = ""

    experiment_config: dict = field(repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dirs = [Path(data_dir) for data_dir in self.data_dirs]

        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)

        if isinstance(self.transforms, list):
            self.transforms = TransformPipelineConfig(self.transforms)

    def save_config(self, path: Path) -> None:
        with path.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(self.experiment_config, f, allow_unicode=True, sort_keys=False)

    def build_transform_compose(self, target: TargetMode | str) -> albu.Compose:
        return self.transforms.to_transform_compose(target)


@dataclass
class Configs:
    experiments: list[ExperimentConfig]

    def __init__(self, experiments: list[dict]) -> None:
        self.experiments = [
            ExperimentConfig(**experiment, experiment_config=experiment)
            for experiment in experiments
        ]


def load_config(config_paths: list[str | Path]) -> Configs:
    raw: dict[str, list[dict[str, Any]]] = {"experiments": []}

    for config_path in config_paths:
        with Path(config_path).open(mode="r", encoding="utf-8") as f:
            raw["experiments"].append(yaml.safe_load(f))

    return Configs(**raw)
