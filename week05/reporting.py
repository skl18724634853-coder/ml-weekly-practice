"""第五周 · 报告组件（reporting.py）

职责：只负责"把已经算好的东西写成文件"——CSV 与 PNG。
**本文件不训练、不评价、不读数据**，只接收现成的记录和指标。

这样拆分的好处：组会上可以明确说"这一段只是保存与绘图，和控制台里训练过程无关"，
绘图代码里不会再偷偷跑一次模型。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只保存图片，不依赖本机图形界面
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


def ensure_results_dir() -> Path:
    """确保 results/ 存在，并返回它的路径。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


# =====================================================================
# CSV
# =====================================================================
def save_step_trace(rows: list[dict], filename: str = "step_trace.csv") -> Path:
    """实践一：保存三行更新记录（清梯度后 / 反传后 / 更新后）。"""
    path = ensure_results_dir() / filename
    pd.DataFrame(rows).round(6).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_history(rows: list[dict], filename: str = "history.csv") -> Path:
    """实践二：保存每轮的训练/验证 MSE，两组配置各 200 行。"""
    path = ensure_results_dir() / filename
    pd.DataFrame(rows).round(6).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_metrics(rows: list[dict], filename: str = "metrics.csv") -> Path:
    """实践二：保存四组模型的指标（原始单位）。"""
    path = ensure_results_dir() / filename
    pd.DataFrame(rows).round(6).to_csv(path, index=False, encoding="utf-8-sig")
    return path


# =====================================================================
# PNG
# =====================================================================
def plot_loss_curves(
    history: pd.DataFrame,
    best_epochs: dict[str, int],
    filename: str = "loss_curves.png",
) -> Path:
    """双面板训练曲线：左 SGD、右 Adam，各自画训练与验证 MSE。

    刻意保留**线性纵轴**（不使用对数轴）：
    这样"验证损失在第几轮开始不再下降、甚至回升"的变化能直接看出来，
    也不会因为对数轴压缩而误以为曲线很平。
    """
    path = ensure_results_dir() / filename
    configs = list(dict.fromkeys(history["config"].tolist()))  # 保持出现顺序
    figure, axes = plt.subplots(1, len(configs), figsize=(13, 5.0), sharex=True)
    if len(configs) == 1:
        axes = [axes]

    for axis, config in zip(axes, configs):
        subset = history[history["config"] == config].sort_values("epoch")
        axis.plot(subset["epoch"], subset["train_MSE"], color="#4C78A8",
                  linewidth=1.9, label="Train MSE")
        axis.plot(subset["epoch"], subset["validation_MSE"], color="#F58518",
                  linewidth=1.9, label="Validation MSE")

        best_epoch = best_epochs.get(config)
        if best_epoch is not None:
            best_row = subset[subset["epoch"] == best_epoch].iloc[0]
            axis.axvline(best_epoch, color="#54A24B", linestyle="--", linewidth=1.4,
                         label=f"Best epoch = {best_epoch}")
            axis.scatter([best_epoch], [best_row["validation_MSE"]],
                         color="#54A24B", s=48, zorder=4)

        axis.set_title(f"{config}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE on standardized target")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    figure.suptitle(
        "Training vs validation MSE (target standardized; linear y-axis)", y=1.01
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_rmse_comparison(
    metrics: pd.DataFrame,
    filename: str = "rmse_comparison.png",
) -> Path:
    """四组模型的训练/验证 RMSE 分组柱状图（原始目标单位）。"""
    path = ensure_results_dir() / filename
    labels = metrics["display_name"].tolist()
    x = range(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.bar([value - width / 2 for value in x], metrics["train_RMSE"], width,
             color="#4C78A8", label="Train RMSE")
    axis.bar([value + width / 2 for value in x], metrics["validation_RMSE"], width,
             color="#F58518", label="Validation RMSE")
    axis.set_xticks(list(x), labels, rotation=10, ha="right")
    axis.set_ylabel("RMSE (original target units)")
    axis.set_title("RMSE comparison: baselines vs MLP (MLP uses its best-epoch weights)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def print_table(title: str, frame: pd.DataFrame) -> None:
    """在控制台打印一张表，便于运行时直接核对（不写文件）。"""
    print(title)
    print(frame.to_string(index=False))
    print()
