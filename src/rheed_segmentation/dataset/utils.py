import itertools
import random
from pathlib import Path

from rheed_segmentation.dataset.label_pair_path import LabelPairPath


def load_paths(root_dir: Path, mask_paths: list[Path], image_type: str) -> list[LabelPairPath]:
    img_paths = []

    for mask_path in mask_paths:
        relative = mask_path.relative_to(root_dir)
        parts = list(relative.parts)
        parts[0] = image_type

        image_path = Path(root_dir, *parts).with_suffix(".tiff")
        if not image_path.exists():
            msg = f"Image not found: {image_path}"
            raise FileNotFoundError(msg)

        img_paths.append(image_path)

    return [LabelPairPath(img, mask) for img, mask in zip(img_paths, mask_paths, strict=True)]


def collect_dataset_paths(
    base_data_dir: Path,
    target_data_dirs: list[str | Path],
    raw_dir_name: str = "raw",
    label_dir_name: str = "label",
) -> list[LabelPairPath]:
    """設定に基づいて、ラベルファイルとそれに対応する画像ファイルのペアのリストを作成します。

    Args:
        base_data_dir: データセット全体のベースディレクトリ ('data'など)。
        target_data_dirs: 収集対象のデータディレクトリ名のリスト。
        raw_dir_name: Raw画像データが格納されているディレクトリ名。
        label_dir_name: ラベルデータが格納されているディレクトリ名。

    Returns:
        LabelPairPath のリスト

    """
    # データファイルパス一覧取得
    all_label_pair_paths = []
    for data_dir in target_data_dirs:
        label_dir = base_data_dir / label_dir_name / data_dir

        # label_dirが存在しない場合はスキップ
        if not label_dir.is_dir():
            print(f"Warning: Directory not found, skipping: {label_dir}")
            continue

        label_paths = sorted(label_dir.glob("**/*.json"))
        label_pair_paths = load_paths(base_data_dir, label_paths, raw_dir_name)
        all_label_pair_paths.append(label_pair_paths)

    # 2次元配列をフラットな1次元配列に変換して返す
    return list(itertools.chain.from_iterable(all_label_pair_paths))


def split_data(
    label_path_pair: list[LabelPairPath], val_ratio: float = 0.2
) -> tuple[list[LabelPairPath], list[LabelPairPath]]:
    random.shuffle(label_path_pair)

    val_size = int(len(label_path_pair) * val_ratio)
    val_paths = label_path_pair[:val_size]
    train_paths = label_path_pair[val_size:]
    return train_paths, val_paths
