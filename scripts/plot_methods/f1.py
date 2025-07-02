import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rheed_segmentation.utils.other import compute_f1_from_confusion_matrix
from rheed_segmentation.utils.result_manager import ResultDateDir


def calculate_moving_average(df: pd.Series, window: int) -> pd.Series:
    return df.rolling(window=window, min_periods=1).mean()


def plot_f1(result_date_dir: ResultDateDir, is_update: bool = False) -> None:
    last_5_f1 = []

    result_protocol_dirs = result_date_dir.fetch_protocol_dirs()
    for result_protocol_dir in result_protocol_dirs:
        name = f"epoch_f1-{result_protocol_dir.protocol}.svg"
        plot_path = result_date_dir.path / name
        if plot_path.exists() and not is_update:
            continue

        # 読み込み
        protocol_df = pd.read_json(result_protocol_dir.history_path, lines=True)

        protocol_df[["f1", "macro-f1"]] = protocol_df["confusion_matrix"].apply(
            lambda x: pd.Series(compute_f1_from_confusion_matrix(np.array(x)))
        )

        data_df = {
            "Spot": calculate_moving_average(protocol_df["f1"].apply(lambda arr: arr[1]), 5),
            "Streak": calculate_moving_average(protocol_df["f1"].apply(lambda arr: arr[2]), 5),
            "Kikuchi": calculate_moving_average(protocol_df["f1"].apply(lambda arr: arr[3]), 5),
            "Macro-F1": calculate_moving_average(protocol_df["macro-f1"], 5),
        }

        last_5_f1.append(
            {
                "protocol": result_protocol_dir.protocol,
                "Spot": float(data_df["Spot"].iloc[-1]),
                "Streak": float(data_df["Streak"].iloc[-1]),
                "Kikuchi": float(data_df["Kikuchi"].iloc[-1]),
                "Macro-F1": float(data_df["Macro-F1"].iloc[-1]),
            }
        )

        # ===== グラフ描画 =====
        dpi = 100
        fig, ax = plt.subplots(figsize=(600 / dpi, 600 / dpi), dpi=dpi)

        ax.set_box_aspect(1)

        # ラベルとフォント
        font_label = {"family": "Arial", "size": 20}
        ax.set_xlabel("Epoch", fontdict=font_label)
        ax.set_ylabel("F1 score", fontdict=font_label)

        # 軸の範囲
        ax.set_xlim(protocol_df["epoch"].min(), protocol_df["epoch"].max())
        ax.set_ylim(0, 1)

        # 目盛り
        ax.tick_params(
            axis="both", which="major", labelsize=12, direction="in", top=True, right=True
        )
        ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)

        # プロット
        plt.rcParams["lines.linewidth"] = 2
        colors = {
            "Spot": "#DD0806",
            "Streak": "#008000",
            "Kikuchi": "#f2bf01",
            "Macro-F1": "#000000",
        }
        for label in colors:  # noqa: PLC0206
            ax.plot(
                protocol_df["epoch"],
                data_df[label],
                label=label,
                color=colors[label],
            )

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

    pd.DataFrame(last_5_f1).to_csv(result_date_dir.path / "last5_f1.csv")
