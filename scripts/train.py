from pathlib import Path

from rheed_segmentation.core import start_experiment

common_config_path = Path("configs", "common.yaml")

config_paths = [
    # Path("configs", "raw.yaml"),
    # Path("configs", "CLAHE.yaml"),
    # Path("configs", "CLAHE_Gaussian.yaml"),
]


config_paths = [
    Path("configs", "weight/CLAHE-1.0.yaml"),
    Path("configs", "weight/CLAHE-0.5.yaml"),
    Path("configs", "weight/CLAHE-0.2.yaml"),
    Path("configs", "weight/CLAHE-0.1.yaml"),
    Path("configs", "weight/CLAHE-0.01.yaml"),
    Path("configs", "weight/CLAHE-10.yaml"),
    Path("configs", "weight/CLAHE-100.yaml"),
    Path("configs", "weight/CLAHE_plot-count.yaml"),
]


def main() -> None:
    start_experiment(config_paths, common_config_path)


if __name__ == "__main__":
    main()
