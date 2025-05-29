import numpy as np
from PIL import Image


def auto_scale(image: Image.Image) -> np.ndarray:
    """PIL画像を自動的にビット深度に基づいて[0, 1]にスケーリング

    Args:
        image (PIL.Image.Image): PIL画像

    Returns:
        np.ndarray: スケーリングされた画像データ ([0, 1] 範囲のNumPy配列)

    """
    bit_depth = get_image_bit_depth(image)
    max_pixel_value: int = 2**bit_depth

    # 正規化
    image_array = np.array(image, dtype=np.float32)
    return image_array / max_pixel_value


def get_image_bit_depth(image: Image.Image) -> int:
    # モードに基づいてビット深度を推定
    mode_to_bit_depth = {
        "L": 8,  # グレースケール
        "P": 8,  # インデックスドカラー
        "I;16": 16,  # 16ビット整数
        "I;16B": 16,  # 16ビット整数
        "I": 32,  # 32ビット整数
        "F": 32,  # 32ビット浮動小数点
    }

    # ビット深度を取得
    return mode_to_bit_depth.get(image.mode, 8)
