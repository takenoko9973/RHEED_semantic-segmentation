import importlib
from types import ModuleType
from typing import Any


def resolve_class(name: str, default_module: ModuleType | None = None) -> Any:  # noqa: ANN401
    """クラス名 `name` からクラスを取得。

    - `name` が `"モジュール.クラス"` の形式ならそれを使用
    - そうでなければ `default_module` から探す

    Args:
        name: クラス名 (例: "Blur", "mypackage.mymodule.classname")
        default_module: デフォルトで探すモジュール (例: albumentations)

    Returns:
        対象のクラスオブジェクト

    """
    if "." in name:
        module_path, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    if default_module is not None:
        return getattr(default_module, name)

    msg = "Please enter a valid class name."
    raise ValueError(msg)
