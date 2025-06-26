from pathlib import Path

from plot_methods.f1 import plot_f1
from plot_methods.loss import plot_loss

from rheed_segmentation.utils.result_manager import ResultDirManager

model_name = "best.pth"

result_root = Path("results")

is_update = True


def main() -> None:
    result_dir_manager = ResultDirManager()
    result_date_dirs = result_dir_manager.get_result_dirs()
    for result_date_dir in result_date_dirs:
        print(result_date_dir)
        plot_loss(result_date_dir, "raw", "CLAHE", is_update)
        plot_f1(result_date_dir, is_update)


if __name__ == "__main__":
    main()
