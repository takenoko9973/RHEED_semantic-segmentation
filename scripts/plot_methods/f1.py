from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rheed_segmentation.metrics import compute_f1_from_confusion_matrix
from rheed_segmentation.utils.result_manager import ResultDateDir


def calculate_moving_average(df: pd.Series, window: int) -> pd.Series:
    return df.rolling(window=window, min_periods=1).mean()


def compute_f1_scores(cm: pd.Series) -> dict[str, pd.Series]:
    protocol_df = pd.DataFrame()
    protocol_df[["f1", "macro-f1"]] = cm.apply(
        lambda x: pd.Series(compute_f1_from_confusion_matrix(np.array(x)))
    )

    return {
        "Spot": calculate_moving_average(protocol_df["f1"].apply(lambda arr: arr[1]), 5),
        "Streak": calculate_moving_average(protocol_df["f1"].apply(lambda arr: arr[2]), 5),
        "Kikuchi": calculate_moving_average(protocol_df["f1"].apply(lambda arr: arr[3]), 5),
        "Macro-F1": calculate_moving_average(protocol_df["macro-f1"], 5),
    }


def _plot_f1_graph(epoch_series: pd.Series, f1_dict: dict[str, pd.Series], plot_path: Path) -> None:
    dpi = 100
    fig, ax = plt.subplots(figsize=(600 / dpi, 600 / dpi), dpi=dpi)

    ax.set_box_aspect(1)

    # ラベルとフォント
    font_label = {"family": "Arial", "size": 20}
    ax.set_xlabel("Epoch", fontdict=font_label)
    ax.set_ylabel("F1 score", fontdict=font_label)

    # 軸の範囲
    ax.set_xlim(epoch_series.min(), epoch_series.max())
    ax.set_ylim(0, 1)

    # 目盛り
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)

    # プロット
    plt.rcParams["lines.linewidth"] = 2
    colors = {
        "Spot": "#DD0806",
        "Streak": "#008000",
        "Kikuchi": "#f2bf01",
        "Macro-F1": "#000000",
    }
    for label, color in colors.items():
        ax.plot(epoch_series, f1_dict[label], label=label, color=color)

    # 凡例
    ax.legend(
        loc="lower right",
        frameon=True,
        handlelength=1.5,
        labelspacing=0.2,
        prop={"family": "Harano Aji Gothic", "size": 16},
    )

    # 保存
    fig.tight_layout()
    plt.savefig(plot_path, format="svg")
    plt.close(fig)  # メモリを解放


def plot_f1(result_date_dir: ResultDateDir, is_update: bool = False) -> None:
    last_5_f1: list[dict] = []

    result_protocol_dirs = result_date_dir.fetch_protocol_dirs()
    for result_protocol_dir in result_protocol_dirs:
        plot_name = f"epoch_f1-{result_protocol_dir.protocol}.svg"
        plot_path = result_date_dir.path / plot_name

        if plot_path.exists() and not is_update:
            continue

        # 読み込み
        protocol_df = pd.read_json(result_protocol_dir.history_path, lines=True)

        f1_dict = compute_f1_scores(protocol_df["confusion_matrix"])
        if "train_confusion_matrix" in protocol_df.columns:
            f1_train_dict = compute_f1_scores(protocol_df["train_confusion_matrix"])
        else:
            f1_train_dict = {}

        last_scores = {"protocol": result_protocol_dir.protocol}
        for label, series in f1_dict.items():
            last_scores[label] = float(series.iloc[-1])
        for label, series in f1_train_dict.items():
            last_scores[f"{label}-train"] = float(series.iloc[-1])
        last_5_f1.append(last_scores)

        _plot_f1_graph(protocol_df["epoch"], f1_dict, plot_path)
        if len(f1_train_dict) != 0:
            _plot_f1_graph(
                protocol_df["epoch"], f1_train_dict, plot_path.with_stem(f"{plot_path.stem}-train")
            )

    last_5_f1_path = result_date_dir.path / "last5_f1.csv"
    pd.DataFrame(last_5_f1).to_csv(last_5_f1_path)
