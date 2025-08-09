from abc import ABC, abstractmethod
from typing import Any

from rheed_segmentation.config.schema import TargetMode


class IConfigAdapter(ABC):
    """古いバージョンの設定辞書を、次のバージョンの辞書形式に変換するためのインターフェース。"""

    @abstractmethod
    def adapt(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """設定辞書を次のバージョンへ変換する。"""


class V1Adapter(IConfigAdapter):
    """バージョン1形式を、バージョン2形式に変換するアダプター。"""

    def adapt(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        print("V1Adapter: v1形式をv2形式へ変換します...")

        # 変換ロジック
        adapted_dict = config_dict.copy()

        # v1の'transform'キーは、実質的に学習時のデータ拡張に相当
        old_transforms: list[dict] = adapted_dict.pop("transforms", [])

        # v2のスキーマ構造に合わせて辞書を再構築
        adapted_dict["transforms"] = {
            "train": {
                "base": [],
                "augmentations": [
                    transform
                    for transform in old_transforms
                    if transform.get("target", TargetMode._BOTH.value)  # noqa: SLF001
                    == TargetMode.TRAIN.value
                ],
                "final": [
                    transform
                    for transform in old_transforms
                    if transform.get("target", TargetMode._BOTH.value)  # noqa: SLF001
                    == TargetMode._BOTH.value  # noqa: SLF001
                ],
            },
            "val": {
                "base": [],
                "augmentations": [],
                "final": [
                    transform
                    for transform in old_transforms
                    if transform.get("target", TargetMode._BOTH.value)  # noqa: SLF001
                    in (TargetMode._BOTH.value, TargetMode.VAL.value)  # noqa: SLF001
                ],
            },
        }
        return adapted_dict


class V2Adapter(IConfigAdapter):
    """バージョン2形式を、バージョン3形式に変換するアダプター。"""

    def adapt(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        print("V2Adapter: v2形式をv3形式へ変換します...")

        # 変換ロジック
        adapted_dict = config_dict.copy()

        # v1の'transform'キーは、実質的に学習時のデータ拡張に相当
        old_transforms: list[dict] = adapted_dict.pop("transforms", [])

        # v2のスキーマ構造に合わせて辞書を再構築
        adapted_dict["transforms"] = {
            "base": old_transforms["train"]["base"],
            "augmentations": old_transforms["train"]["augmentations"],
            "final": old_transforms["train"]["final"],
        }
        return adapted_dict
