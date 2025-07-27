from typing import Any

import albumentations as albu
from albumentations.core.transforms_interface import BasicTransform
from pydantic import Field

from rheed_segmentation.utils import resolve_class

from .core import BaseConfig


class Transform(BaseConfig):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def build_transform(self) -> BasicTransform:
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


class TransformGroup(BaseConfig):
    base: list[Transform] = Field(default_factory=list)
    augmentations: list[Transform] = Field(default_factory=list)
    final: list[Transform] = Field(default_factory=list)

    def build_transform_compose(self) -> albu.Compose:
        base_transforms = [trans.build_transform() for trans in self.base]
        aug_transforms = [trans.build_transform() for trans in self.augmentations]
        final_transforms = [trans.build_transform() for trans in self.final]
        return albu.Compose([*base_transforms, *aug_transforms, *final_transforms])


class TransformPipelines(BaseConfig):
    train: TransformGroup
    val: TransformGroup

    def build_train_trainsform_compose(self) -> albu.Compose:
        return self.train.build_transform_compose()

    def build_val_trainsform_compose(self) -> albu.Compose:
        return self.val.build_transform_compose()
