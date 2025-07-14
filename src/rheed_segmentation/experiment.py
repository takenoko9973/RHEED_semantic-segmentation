from torch.utils.data import DataLoader

from rheed_segmentation.config import Configs, ExperimentConfig
from rheed_segmentation.config.transform_config import TargetMode
from rheed_segmentation.dataset import obtain_datasets
from rheed_segmentation.train import Trainer
from rheed_segmentation.utils.other import init_random_seed
from rheed_segmentation.utils.result_manager import ResultDateDir, ResultDirManager


def training_experiment(
    experiment_config: ExperimentConfig, result_date_dir: ResultDateDir
) -> None:
    init_random_seed(917)

    print(f"protocol: {experiment_config.protocol}, comment: {experiment_config.comment}")

    # 学習モデル保存先作成
    result_dir = result_date_dir.create_protocol_dir(protocol=experiment_config.protocol)

    # 設定保存
    experiment_config.save_config(result_dir.path / "config.yaml")

    # データ取得
    train_transform = experiment_config.build_transform_compose(TargetMode.TRAIN)
    val_transform = experiment_config.build_transform_compose(TargetMode.VAL)
    train_dataset, val_dataset = obtain_datasets(experiment_config, train_transform, val_transform)
    train_dataset.save_dataset_list(result_dir.path / "train_list.txt")
    val_dataset.save_dataset_list(result_dir.path / "val_list.txt")

    # 学習
    batch_size: int = experiment_config.training.batch_size
    num_workers: int = experiment_config.training.num_workers
    trainer = Trainer(
        experiment_config.training,
        len(experiment_config.labels),
        DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers),
        result_dir,
    )
    trainer.train(experiment_config.training.epoch)


def training_experiments(configs: Configs) -> None:
    result_manager = ResultDirManager()

    result_date_dir = result_manager.create_date_dir(configs.common_name)
    configs.save_common_config(result_date_dir.path / "common_config.yaml")

    for experiment in configs.experiments:
        training_experiment(experiment, result_date_dir)
