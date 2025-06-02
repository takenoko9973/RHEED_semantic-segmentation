from enum import Enum


def enum_or_default[T: Enum](value: str | T, enum_cls: type[T], default: T) -> T:
    if isinstance(value, enum_cls):
        return value

    try:
        return enum_cls(value)
    except ValueError:
        print(f"[Warn] 不正な値 '{value}'。{default} にフォールバックします。")
        return default
