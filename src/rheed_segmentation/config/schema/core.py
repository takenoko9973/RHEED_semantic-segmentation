from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """全てのPydantic設定モデルが継承するベースクラス。共通のモデル挙動 (ConfigDict) を定義する。"""

    model_config = ConfigDict(
        populate_by_name=True,  # エイリアス名でのデータ入力許可
        arbitrary_types_allowed=True,  # PyTorchオブジェクトなどの任意の型を許容
    )
