from enum import Enum
from typing import Any

import albumentations as albu
from albumentations.core.transforms_interface import BasicTransform
from pydantic import BaseModel, Field, field_validator

from rheed_segmentation.utils import enum_or_default, resolve_class


class TargetMode(Enum):
    TRAIN = "train"
    VAL = "val"
    BOTH = "both"


class TransformConfig(BaseModel):
    """画像変換処理の設定を定義"""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    target: TargetMode = TargetMode.BOTH

    @classmethod
    @field_validator("target")
    def _target_mode(cls, value: Any) -> TargetMode:  # noqa: ANN401
        enum_or_default(value, TargetMode, TargetMode.BOTH)

    def to_transform(self) -> BasicTransform:
        """設定に基づいて albumentations の変換オブジェクトを生成"""
        try:
            transform_cls = resolve_class(self.name, default_module=albu)
            return transform_cls(**self.params)
        except (ImportError, AttributeError, TypeError) as e:
            msg = (
                f"Failed to resolve or instantiate transform '{self.name}' "
                f"with params {self.params}."
            )
            raise ValueError(msg) from e


class TransformPipelineConfig(BaseModel):
    """変換処理のパイプライン全体を定義"""

    transform_configs: list[TransformConfig] = Field(default_factory=list)

    @classmethod
    @field_validator("transform_configs")
    def _transform_configs_trans(cls, value: Any) -> TargetMode:  # noqa: ANN401
        return [
            TransformConfig(**targetform_config)
            for targetform_config in value
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


if __name__ == "__main__":
    transforms = [
        {"name": "Resize", "params": {"height": 135, "width": 180}},
        {"name": "ToTensorV2"},
    ]

    transforms_config = TransformPipelineConfig(transform_configs=transforms)
    print(transforms_config)
