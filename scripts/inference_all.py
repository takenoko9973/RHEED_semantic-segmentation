from pathlib import Path

from rheed_segmentation.config.experiment_config import ExperimentConfig
from rheed_segmentation.dataset import collect_dataset_paths
from rheed_segmentation.utils.result_manager import ResultDir, ResultDirManager
from rheed_segmentation.visualization.inference import inference
from rheed_segmentation.visualization.model import load_model
from rheed_segmentation.visualization.visualize import save_prediction

model_name = "best.pth"

result_root = Path("results")
data_root = Path("data")
image_root = data_root / "raw"

result_dir_name = "20250708215052-STO-0,10_MgO-0_YBCO-STO-0"


def inference_all(result_protocol_dir: ResultDir) -> None:
    config_path = result_protocol_dir.path / "config.yaml"
    model_path = result_protocol_dir.path / model_name

    try:
        config = ExperimentConfig.from_path(config_path)
        model = load_model(model_path, config.training)
        pair_paths = collect_dataset_paths(Path("data"), config.data_dirs)
    except FileNotFoundError:
        print(f"読み込み時にエラーが発生しました {result_protocol_dir}")
        return

    base_save_path = Path(result_protocol_dir.path, "preds_all")
    for pair_path in pair_paths:
        save_path = base_save_path / pair_path.image_path.relative_to(image_root)
        if save_path.exists():
            continue

        pred = inference(config, model, pair_path.image_path)
        save_prediction(pred, save_path)


def main() -> None:
    result_dir_manager = ResultDirManager()
    result_date_dir = result_dir_manager.get_result_dir_from_name(result_dir_name)

    for result_protocol_dir in result_date_dir.fetch_protocol_dirs():
        print(result_protocol_dir)

        inference_all(result_protocol_dir)


if __name__ == "__main__":
    main()
