from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import albumentations as albu
from albumentations.core.transforms_interface import BasicTransform

from rheed_segmentation.utils import enum_or_default, resolve_class


class TargetMode(Enum):
    TRAIN = "train"
    VAL = "val"
    BOTH = "both"


@dataclass
class TransformConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    target: TargetMode = TargetMode.BOTH

    def __post_init__(self) -> None:
        self.target = enum_or_default(self.target, TargetMode, TargetMode.BOTH)

    def to_transform(self) -> BasicTransform:
        cls = resolve_class(self.name, albu)
        return cls(**self.params)


@dataclass
class TransformPipelineConfig:
    transform_configs: list[TransformConfig]

    def __post_init__(self) -> None:
        self.transform_configs = [
            TransformConfig(**targetform_config)
            for targetform_config in self.transform_configs
            if isinstance(targetform_config, dict)
        ]

    def to_transform_compose(self, target: TargetMode | str = TargetMode.TRAIN) -> albu.Compose:
        if target not in TargetMode:
            target = TargetMode.TRAIN

        transforms = [
            cfg.to_transform()
            for cfg in self.transform_configs
            if cfg.target in (TargetMode.BOTH, target)
        ]
        return albu.Compose([*transforms])
