from pathlib import Path
from typing import Any

import yaml

from .adapter import IConfigAdapter, V1Adapter
from .schema import ProtocolConfig

ADAPTER_MAP: dict[int, IConfigAdapter] = {
    1: V1Adapter(),
}
LATEST_VERSION = 2


def _merge_dicts(base: dict, override: dict) -> dict:
    base_version = base.get("version", 1)
    override_version = base.get("version", 1)

    if base_version != override_version:
        msg = "マージする設定のバージョンが合いません。"
        raise ValueError(msg)

    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _merge_dicts(result[key], val)
        else:
            result[key] = val
    return result


def _adapt_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """バージョンを判定し、必要なら変換をかけてPydanticモデルを返す"""
    current_version = config_dict.get("version", 1)

    pipeline = [ADAPTER_MAP[v] for v in range(current_version, LATEST_VERSION)]

    transformed_dict = config_dict
    for adapter in pipeline:
        transformed_dict = adapter.adapt(transformed_dict)  # adaptメソッドで変換

    return transformed_dict


def load_config(config_path: Path, common_config_path: Path | None = None) -> ProtocolConfig:
    with config_path.open("r", encoding="utf-8") as f:
        experiment_dict = yaml.safe_load(f)

    merged_dict = experiment_dict
    if common_config_path and common_config_path.exists():
        with common_config_path.open("r", encoding="utf-8") as f:
            common_dict = yaml.safe_load(f)
        merged_dict = _merge_dicts(common_dict, experiment_dict)

    transformed_dict = _adapt_config(merged_dict)
    return ProtocolConfig.model_validate(transformed_dict)


def load_multiple_configs(
    config_paths: list[Path], common_config_path: Path | None = None
) -> list[ProtocolConfig]:
    return [load_config(p, common_config_path) for p in config_paths]
