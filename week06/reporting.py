"""第六周 · CSV 保存与三张要求图的绘制。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """统一保存 UTF-8 CSV；父目录不存在时自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def plot_tuning_comparison(tuning: pd.DataFrame, path: Path) -> None:
    """画训练、验证 RMSE 柱状对照图；横轴是不同配置，不是训练轮次。"""
    labels: list[str] = []
    for _, row in tuning.iterrows():
        if row["model"] == "lag_linear":
            labels.append("lag_linear")
        else:
            labels.append(f"ridge\nα={row['param_value']:g}")

    positions = np.arange(len(tuning))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(positions - width / 2, tuning["train_rmse"], width, label="Train RMSE", color="#4c9ed9")
    ax.bar(positions + width / 2, tuning["val_rmse"], width, label="Validation RMSE", color="#f2a65a")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("RMSE (OT original scale)")
    ax.set_title("Tuning comparison: train vs validation RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_series_split(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    initial_history_end: pd.Timestamp,
    path: Path,
) -> None:
    """画前 14,400 小时油温，并标出验证/测试边界和初始历史窗口。"""
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(frame["date"], frame["OT"], color="#1d77a5", linewidth=0.8, label="OT")
    ax.axvspan(frame["date"].iloc[0], initial_history_end, color="#f4cf65", alpha=0.35, label="initial 24-hour history")
    ax.axvline(validation_start, color="#e27d60", linestyle="--", linewidth=1.8, label="validation start")
    ax.axvline(test_start, color="#8c5fb5", linestyle="--", linewidth=1.8, label="test start")
    ax.set_title("ETTh1 oil temperature and chronological splits")
    ax.set_xlabel("Date")
    ax.set_ylabel("Oil temperature OT (original data scale)")
    ax.legend(ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_validation_predictions(
    predictions: pd.DataFrame,
    raw_frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    path: Path,
) -> None:
    """画验证首周：上方是真实 OT 和四种预测，下方是同时间的 HUFL。"""
    week_end = validation_start + pd.Timedelta(days=7)
    validation = predictions[
        (predictions["split"] == "validation")
        & (predictions["target_time"] >= validation_start)
        & (predictions["target_time"] < week_end)
    ].copy()
    pivot = validation.pivot(index="target_time", columns="model", values="y_pred").sort_index()
    truth = validation.groupby("target_time", sort=True)["y_true"].first()
    load = (
        raw_frame[(raw_frame["date"] >= validation_start) & (raw_frame["date"] < week_end)]
        .set_index("date")["HUFL"]
        .reindex(truth.index)
    )

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=(2.2, 1))
    top.plot(truth.index, truth.values, color="#111827", linewidth=2.2, label="True OT")
    colors = {"last_value": "#e76f51", "window_mean": "#e9c46a", "lag_linear": "#2a9d8f", "ridge_tuned": "#457b9d"}
    for name in pivot.columns:
        top.plot(pivot.index, pivot[name], linewidth=1.2, label=name, color=colors.get(name))
    top.set_ylabel("Oil temperature OT\n(original data scale)")
    top.set_title("Validation first week: oil temperature predictions and HUFL")
    top.legend(ncol=3, fontsize=9)
    top.grid(alpha=0.25)

    bottom.plot(load.index, load.values, color="#7b2cbf", linewidth=1.2)
    bottom.set_ylabel("HUFL\n(original data scale)")
    bottom.set_xlabel("Date")
    bottom.grid(alpha=0.25)
    bottom.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
