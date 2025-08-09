from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from rheed_segmentation.config import ProtocolConfig, TargetMode, load_multiple_configs
from rheed_segmentation.dataset import obtain_datasets
from rheed_segmentation.train import Trainer
from rheed_segmentation.utils.other import init_random_seed
from rheed_segmentation.utils.result_manager import ResultDateDir, ResultDirManager


def _save_common_config(common_config_path: Path, result_date_dir: ResultDateDir) -> None:
    """共有設定ファイルの保存"""
    if common_config_path and common_config_path.exists():
        common_config_save_path = result_date_dir.path / "common_config.yaml"
        with (
            common_config_path.open("r", encoding="utf-8") as common_config_file,
            common_config_save_path.open("w", encoding="utf-8") as save_file,
        ):
            common_config_dict = yaml.safe_load(common_config_file)
            yaml.safe_dump(common_config_dict, save_file, allow_unicode=True)
            del common_config_dict


def training_protocol(protocol_config: ProtocolConfig, result_date_dir: ResultDateDir) -> None:
    print(f"protocol: {protocol_config.protocol}, comment: {protocol_config.comment}")

    # トレーニング開始前に乱数リセット
    init_random_seed(917)

    # 学習モデル保存先作成
    result_dir = result_date_dir.create_protocol_dir(protocol=protocol_config.protocol)

    # 設定保存
    protocol_config.save_config(result_dir.path / "config.yaml")

    # データ取得
    train_transform = protocol_config.build_transform_compose(TargetMode.TRAIN)
    val_transform = protocol_config.build_transform_compose(TargetMode.VAL)
    train_dataset, val_dataset = obtain_datasets(protocol_config, train_transform, val_transform)
    train_dataset.save_dataset_list(result_dir.path / "train_list.txt")
    val_dataset.save_dataset_list(result_dir.path / "val_list.txt")

    # 学習
    batch_size: int = protocol_config.training.batch_size
    num_workers: int = protocol_config.training.num_workers
    trainer = Trainer(
        protocol_config.training,
        len(protocol_config.labels),
        DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers),
        result_dir,
    )
    trainer.train(protocol_config.training.epoch)


def start_experiment(
    config_paths: list[Path],
    common_config_path: Path | None = None,
) -> None:
    if len(config_paths) == 0:
        msg = "1つ以上の設定ファイルを入力してください"
        raise ValueError(msg)

    protocol_configs = load_multiple_configs(config_paths, common_config_path)

    result_manager = ResultDirManager()
    result_date_dir = result_manager.create_date_dir(protocol_configs[0].common_name)

    _save_common_config(common_config_path, result_date_dir)

    for protocol_config in protocol_configs:
        training_protocol(protocol_config, result_date_dir)
