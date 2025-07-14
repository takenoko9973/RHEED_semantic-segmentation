import json
from pathlib import Path

import numpy as np
import torch

from rheed_segmentation.config.experiment_config import ExperimentConfig
from rheed_segmentation.config.transform_config import TargetMode
from rheed_segmentation.dataset.dataset import SegmentationDataset
from rheed_segmentation.dataset.factory import obtain_datasets_from_paths
from rheed_segmentation.dataset.finder import load_paths
from rheed_segmentation.dataset.path import LabelPairPath
from rheed_segmentation.metrics import compute_confusion_matrix, compute_f1_from_confusion_matrix
from rheed_segmentation.utils.result_manager import ResultDateDir, ResultDir, ResultDirManager
from rheed_segmentation.visualization.model import load_model

model_name = "best.pth"

result_root = Path("results")

is_update = True


def _obtain_path_pairs(
    result_protocol_dir: ResultDir,
) -> tuple[list[LabelPairPath], list[LabelPairPath]]:
    train_json_list_path = result_protocol_dir.path / "train_list.txt"
    val_json_list_path = result_protocol_dir.path / "val_list.txt"
    train_json_list = []
    val_json_list = []

    if train_json_list_path.exists():
        with train_json_list_path.open() as f:
            train_json_list = [Path(line.strip()) for line in f.readlines()]
    if val_json_list_path.exists():
        with val_json_list_path.open() as f:
            val_json_list = [Path(line.strip()) for line in f.readlines()]

    train_path_pairs = load_paths(Path("data"), train_json_list, "raw")
    val_path_pairs = load_paths(Path("data"), val_json_list, "raw")
    return train_path_pairs, val_path_pairs


def compute_each_type_f1_score(
    config: ExperimentConfig, model: torch.nn.Module, dataset: SegmentationDataset, save_path: Path
) -> None:
    if save_path.exists():
        return
    if len(dataset) == 0:
        return

    num_classes = len(config.labels)
    data_list = {
        str(data_dir): np.zeros((num_classes, num_classes), dtype=np.int64)
        for data_dir in config.data_dirs
    }

    #  予測
    for index in range(len(dataset)):
        img, mask = dataset[index]
        data_path = dataset.label_pair_paths[index].json_path.parents[1]
        data_type = "/".join(data_path.parts[2:])

        with torch.no_grad():
            pred = model(img.unsqueeze(0))

        data_list[data_type] += compute_confusion_matrix(pred, mask.long(), num_classes)

    # 各データ(試料)に対する f1 計算
    f1_each_type = {}
    for key, cm in data_list.items():
        f1, _ = compute_f1_from_confusion_matrix(cm)
        f1_each_type[key] = f1.tolist()

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(f1_each_type, f)


def each_type_f1_score(result_date_dir: ResultDateDir) -> None:
    result_protocol_dirs = result_date_dir.fetch_protocol_dirs()
    for result_protocol_dir in result_protocol_dirs:
        protocol_path = result_protocol_dir.path
        config_path = protocol_path / "config.yaml"
        model_path = protocol_path / model_name

        try:
            config = ExperimentConfig.from_path(config_path)

            transform = config.build_transform_compose(TargetMode.VAL)
            train_path_pairs, val_path_pairs = _obtain_path_pairs(result_protocol_dir)
            if len(train_path_pairs) + len(val_path_pairs) == 0:
                print("skip")
                continue

            train_dataset = obtain_datasets_from_paths(config, train_path_pairs, transform)
            val_dataset = obtain_datasets_from_paths(config, val_path_pairs, transform)

            model = load_model(model_path, config.training)

        except FileNotFoundError:
            print("読み込み時にエラーが発生しました")
            continue

        print("loaded")

        compute_each_type_f1_score(
            config,
            model,
            train_dataset,
            protocol_path.parent / f"each_f1-{result_protocol_dir.protocol}_train.json",
        )
        compute_each_type_f1_score(
            config,
            model,
            val_dataset,
            protocol_path.parent / f"each_f1-{result_protocol_dir.protocol}.json",
        )

        print("computed")


def main() -> None:
    result_dir_manager = ResultDirManager()
    result_date_dirs = result_dir_manager.get_result_dirs()
    for result_date_dir in result_date_dirs:
        print(result_date_dir)
        each_type_f1_score(result_date_dir)


if __name__ == "__main__":
    main()
