from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rheed_segmentation.utils.other import compute_f1_from_confusion_matrix
from rheed_segmentation.utils.result_manager import ResultDateDir, ResultDirManager

model_name = "best.pth"

result_root = Path("results")


def plot_loss(result_date_dir: ResultDateDir, raw_protocol: str, adjusted_protocol: str) -> None:
    raw_dir = result_date_dir.fetch_protocol_dir_by_name(raw_protocol)
    adjusted_dir = result_date_dir.fetch_protocol_dir_by_name(adjusted_protocol)

    if raw_dir is None or adjusted_dir is None:
        return

    raw_df = pd.read_json(raw_dir.history_path, lines=True)
    clahe_df = pd.read_json(adjusted_dir.history_path, lines=True)

    dpi = 100
    fig, ax = plt.subplots(figsize=(600 / dpi, 600 / dpi), dpi=dpi)

    ax.set_box_aspect(1)

    # 軸
    ax.set_xscale("linear")
    ax.set_yscale("log")

    # 軸ラベル
    ax.set_xlabel("Epoch", fontsize=20, fontname="Arial")
    ax.set_ylabel("Loss", fontsize=20, fontname="Arial")

    # 範囲
    ax.set_xlim(raw_df["epoch"].min(), raw_df["epoch"].max())
    ax.set_ylim(1e-2, 1e1)

    # 目盛りフォント
    ax.tick_params(axis="both", which="major", labelsize=15, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)

    # 対数目盛形式 (10^{x})
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: rf"$10^{{{np.log10(y):.0f}}}$"))

    # プロット
    ax.plot(
        raw_df["epoch"],
        raw_df["train_loss"],
        label="train loss",
        color="black",
        linestyle="dashed",
        linewidth=2,
    )
    ax.plot(
        raw_df["epoch"],
        raw_df["validate_loss"],
        label="val loss",
        color="black",
        linestyle="-",
        linewidth=2,
    )
    ax.plot(
        clahe_df["epoch"],
        clahe_df["train_loss"],
        label="train loss (adjusted)",
        color="red",
        linestyle="dashed",
        linewidth=2,
    )
    ax.plot(
        clahe_df["epoch"],
        clahe_df["validate_loss"],
        label="val loss (adjusted)",
        color="red",
        linestyle="-",
        linewidth=2,
    )

    # 凡例
    ax.legend(
        loc="upper right",
        frameon=True,
        handlelength=2,
        borderpad=0.5,
        labelspacing=0.5,
        prop={"family": "Harano Aji Gothic", "size": 16},
    )

    # グラフ形状の調整 (正方形にする)
    # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    # 出力
    fig.tight_layout()
    name = f"epoch_loss-{raw_protocol}-{adjusted_protocol}.svg"
    path = result_root / result_date_dir.date_str / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_f1(result_date_dir: ResultDateDir) -> None:
    for result_protocol_dir in result_date_dir.fetch_protocol_dirs():
        protocol_df = pd.read_json(result_protocol_dir.history_path, lines=True)

        protocol_df[["f1", "macro-f1"]] = protocol_df["confusion_matrix"].apply(
            lambda x: pd.Series(compute_f1_from_confusion_matrix(np.array(x)))
        )

        # ===== グラフ描画 =====
        dpi = 100
        fig, ax = plt.subplots(figsize=(600 / dpi, 480 / dpi), dpi=dpi)

        ax.set_box_aspect(1)

        # ラベルとフォント
        font_label = {"family": "Arial", "size": 20}
        ax.set_xlabel("Epoch", fontdict=font_label)
        ax.set_ylabel("F1 score", fontdict=font_label)

        # 軸の範囲
        ax.set_xlim(protocol_df["epoch"].min(), protocol_df["epoch"].max())
        ax.set_ylim(0, 1)

        # プロット
        colors = {
            "Spot": "#DD0806",
            "Streak": "#0000D4",
            "Kikuchi": "#07601A",
            "Macro-F1": "#000000",
        }
        ax.plot(
            protocol_df["epoch"],
            protocol_df["f1"].apply(lambda arr: arr[1]),
            label="Spot",
            color=colors["Spot"],
            linewidth=2,
        )
        ax.plot(
            protocol_df["epoch"],
            protocol_df["f1"].apply(lambda arr: arr[2]),
            label="Streak",
            color=colors["Streak"],
            linewidth=2,
        )
        ax.plot(
            protocol_df["epoch"],
            protocol_df["f1"].apply(lambda arr: arr[3]),
            label="Kikuchi",
            color=colors["Kikuchi"],
            linewidth=2,
        )
        ax.plot(
            protocol_df["epoch"],
            protocol_df["macro-f1"],
            label="Macro-F1",
            color=colors["Macro-F1"],
            linewidth=2,
        )

        # 凡例
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles[::-1],
            labels[::-1],  # key reverse
            loc="lower left",  # key at ... は bbox_to_anchorとlocで指定
            bbox_to_anchor=(280, 0.3),  # key at 280,0.3 (データ座標)
            prop={"family": "Harano Aji Gothic", "size": 16},
            frameon=True,  # key box
            handlelength=1.8,
        )  # key width -1 (デフォルトより少し短く)
        legend.get_frame().set_edgecolor("black")

        # 目盛り
        font_ticks = {"family": "Arial", "size": 15}
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=font_ticks["size"],
            direction="in",
            top=True,
            right=True,
        )  # xtics/ytics mirror
        ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(font_ticks["family"])

        # 保存
        fig.tight_layout()
        name = f"epoch_f1-{result_protocol_dir.protocol}.svg"
        path = result_root / result_date_dir.date_str / name
        plt.savefig(path, format="svg")
        plt.close(fig)  # メモリを解放


def main() -> None:
    result_dir_manager = ResultDirManager()
    result_date_dirs = result_dir_manager.fetch_result_dirs()
    for result_date_dir in result_date_dirs:
        plot_loss(result_date_dir, "raw", "CLAHE")
        plot_f1(result_date_dir)


if __name__ == "__main__":
    main()
