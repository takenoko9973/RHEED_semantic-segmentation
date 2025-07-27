from enum import Enum
from pathlib import Path

import albumentations as albu
import yaml
from pydantic import Field

from .core import BaseConfig
from .training import TrainingConfig
from .transform import TransformPipelines


class TargetMode(Enum):
    TRAIN = "train"
    VAL = "val"
    _BOTH = "both"


class ProtocolConfig(BaseConfig):
    protocol: str
    data_dirs: list[Path]
    labels: dict[str, int]
    per_label: bool
    training: TrainingConfig
    transforms: TransformPipelines
    common_name: str = ""
    comment: str = ""
    version: int = Field(default=1, validate_default=True)

    def build_transform_compose(self, target: TargetMode | str) -> albu.Compose:
        if target == TargetMode.TRAIN:
            return self.transforms.build_train_trainsform_compose()
        if target == TargetMode.VAL:
            return self.transforms.build_val_trainsform_compose()

        return None

    def save_config(self, path: Path) -> None:
        config_dict = self.model_dump(mode="json", by_alias=True, exclude_none=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)
