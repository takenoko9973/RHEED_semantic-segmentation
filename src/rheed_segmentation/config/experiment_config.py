from pathlib import Path
from typing import Any

import albumentations as albu
import yaml
from pydantic import Field, field_validator, model_validator

from .core import BaseConfig
from .training_config import TrainingConfig
from .transform_config import TargetMode, TransformPipelineConfig


class ExperimentConfig(BaseConfig):
    protocol: str
    data_dirs: list[Path]
    labels: dict[str, int]
    per_label: bool
    training: TrainingConfig
    transforms: TransformPipelineConfig
    common_name: str = ""
    comment: str = ""

    @field_validator("transforms", mode="before")
    @classmethod
    def _wrap_transforms_in_dict(cls, v: Any) -> Any:  # noqa: ANN401
        """YAMLのリストをTransformPipelineConfigが解釈できる辞書に変換する"""
        if isinstance(v, list):
            return {"transform_configs": v}
        return v

    def build_transform_compose(self, target: TargetMode | str) -> albu.Compose:
        return self.transforms.to_transform_compose(target)

    def save_config(self, path: Path) -> None:
        config_dict = self.model_dump(mode="json", by_alias=True, exclude_none=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)


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


class Configs(BaseConfig):
    experiments: list[ExperimentConfig]
    common_config: dict = Field(repr=False, default_factory=dict)
    common_name: str = ""

    @model_validator(mode="before")
    @classmethod
    def from_paths(cls, data: dict) -> dict[str, Any]:
        """ファイルパスの辞書からモデルを構築するためのメインロジック。"""
        if not isinstance(data, dict):
            msg = "Initialization data for ConfigsPydantic must be a dictionary."
            raise TypeError(msg)

        config_paths = data.get("config_paths", [])
        common_config_path = data.get("common_config_path")

        # 共通設定ファイル読み込み
        common_dict = {}
        if common_config_path and Path(common_config_path).exists():
            with Path(common_config_path).open(mode="r", encoding="utf-8") as f:
                common_dict = yaml.safe_load(f)

        # 各設定ファイルを読み込み、共通設定とマージ
        loaded_experiments = []
        for config_path in config_paths:
            with Path(config_path).open(mode="r", encoding="utf-8") as f:
                experiment_dict = yaml.safe_load(f)

            merged_dict = merge_dicts(common_dict, experiment_dict)
            loaded_experiments.append(merged_dict)

        # モデルのフィールドに合わせた辞書を返却する
        return {
            "experiments": loaded_experiments,
            "common_config": common_dict,
            "common_name": common_dict.get("common_name", ""),
        }

    def save_common_config(self, path: Path) -> None:
        with path.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(self.common_config, f, allow_unicode=True)
