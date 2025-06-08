from rheed_segmentation.config import Configs, ExperimentConfig
from rheed_segmentation.config.transform_config import TargetMode
from rheed_segmentation.data_loader import make_dataloaders
from rheed_segmentation.train import Trainer
from rheed_segmentation.utils.other import init_random_seed
from rheed_segmentation.utils.result_manager import ResultDirManager


def training_experiment(experiment_config: ExperimentConfig) -> None:
    init_random_seed(917)

    result_manager = ResultDirManager()
    result_dir = result_manager.create_result_dir(protocol=experiment_config.protocol)

    experiment_config.save_config(result_dir.path / "config.yaml")

    train_transform = experiment_config.build_transform_compose(TargetMode.TRAIN)
    val_transform = experiment_config.build_transform_compose(TargetMode.VAL)
    train_loader, val_loader = make_dataloaders(experiment_config, train_transform, val_transform)

    trainer = Trainer(
        experiment_config.training,
        len(experiment_config.labels),
        train_loader,
        val_loader,
        result_dir,
    )
    trainer.train(experiment_config.training.epoch)


def training_experiments(config: Configs) -> None:
    for experiment in config.experiments:
        training_experiment(experiment)
