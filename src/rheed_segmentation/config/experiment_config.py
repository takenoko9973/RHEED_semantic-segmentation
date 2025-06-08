from dataclasses import dataclass, field
from pathlib import Path

import albumentations as albu
import yaml

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
            self.transforms = TransformPipelineConfig(transform_configs=self.transforms)

    def save_config(self, path: Path) -> None:
        with path.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(self.experiment_config, f, allow_unicode=True)

    def build_transform_compose(self, target: TargetMode | str) -> albu.Compose:
        return self.transforms.to_transform_compose(target)


@dataclass
class Configs:
    experiments: list[ExperimentConfig]

    def __post_init__(self) -> None:
        self.experiments = [
            ExperimentConfig(**experiment, experiment_config=experiment)
            if isinstance(experiment, dict)
            else experiment
            for experiment in self.experiments
        ]


def merge_dicts(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key == "comment":
            # comment は結合
            base_comment: str = base.get("comment", "")
            override_comment: str = val

            if base_comment and override_comment:
                result["comment"] = base_comment.rstrip() + " / " + override_comment.lstrip()
            else:
                result["comment"] = base_comment or override_comment

        elif key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_dicts(result[key], val)
        else:
            result[key] = val

    return result


def load_config(
    config_path: str | Path, common_config_path: str | Path | None = None
) -> ExperimentConfig:
    if common_config_path is not None:
        with Path(common_config_path).open(mode="r", encoding="utf-8") as f:
            common_dict = yaml.safe_load(f)

    with Path(config_path).open(mode="r", encoding="utf-8") as f:
        experiment_dict = yaml.safe_load(f)

    if common_config_path is not None:
        experiment_dict = merge_dicts(common_dict, experiment_dict)

    return ExperimentConfig(**experiment_dict, experiment_config=experiment_dict)


def load_configs(
    config_paths: list[str | Path], common_config_path: str | Path | None = None
) -> Configs:
    experiment_configs = [
        load_config(config_path, common_config_path) for config_path in config_paths
    ]

    return Configs(experiment_configs)
