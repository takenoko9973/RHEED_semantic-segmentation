import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rheed_segmentation.utils.result_manager import ResultDateDir


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
    font_label = {"family": "Arial", "size": 20}
    ax.set_xlabel("Epoch", fontdict=font_label)
    ax.set_ylabel("Loss", fontdict=font_label)

    # 範囲
    ax.set_xlim(raw_df["epoch"].min(), raw_df["epoch"].max())
    ax.set_ylim(1e-2, 1e1)

    # 目盛りフォント
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True)

    # 対数目盛形式 (10^{x})
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: rf"$10^{{{np.log10(y):.0f}}}$"))

    # プロット
    plt.rcParams["lines.linewidth"] = 2
    ax.plot(
        raw_df["epoch"], raw_df["train_loss"], label="train loss", color="black", linestyle="--"
    )
    ax.plot(
        raw_df["epoch"], raw_df["validate_loss"], label="val loss", color="black", linestyle="-"
    )
    ax.plot(
        clahe_df["epoch"],
        clahe_df["train_loss"],
        label="train loss (adjusted)",
        color="red",
        linestyle="--",
    )
    ax.plot(
        clahe_df["epoch"],
        clahe_df["validate_loss"],
        label="val loss (adjusted)",
        color="red",
        linestyle="-",
    )

    # 凡例
    ax.legend(
        loc="upper right",
        frameon=True,
        handlelength=1.5,
        labelspacing=0.2,
        prop={"family": "Harano Aji Gothic", "size": 16},
    )

    # グラフ形状の調整 (正方形にする)
    # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    # 出力
    fig.tight_layout()
    name = f"epoch_loss-{raw_protocol}-{adjusted_protocol}.svg"
    path = result_date_dir.path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
