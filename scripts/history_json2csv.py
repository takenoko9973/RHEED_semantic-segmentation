import json
from pathlib import Path

import numpy as np
import pandas as pd

from rheed_segmentation.utils import compute_f1_from_confusion_matrix

result_root = Path("results")


def convert(experiment_path: Path) -> None:
    history_path = experiment_path / "history.jsonl"

    with history_path.open() as f:
        history_list = [json.loads(line) for line in f.readlines()]

    history_list2 = []
    for history in history_list:
        history2 = {}

        history2["epoch"] = history["epoch"]
        history2["train_loss"] = history["train_loss"]
        history2["validate_loss"] = history["validate_loss"]

        cm = np.array(history["confusion_matrix"])

        f1_scores, macro_f1 = compute_f1_from_confusion_matrix(cm)

        history2["spot_f1"] = f1_scores[2]
        history2["streak_f1"] = f1_scores[3]
        history2["kikuchi_f1"] = f1_scores[1]
        history2["macro_f1"] = macro_f1

        history_list2.append(history2)

    pd.DataFrame(history_list2).to_csv(history_path.with_suffix(".csv"))


def main() -> None:
    for experiment_path in result_root.glob("*/"):
        convert(experiment_path)


if __name__ == "__main__":
    main()
