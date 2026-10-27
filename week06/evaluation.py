"""第六周 · 回归评价、汇总指标与逐条预测表。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """在原始油温尺度上计算 MAE 与 RMSE。"""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != prediction.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致：{actual.shape} vs {prediction.shape}")
    error = prediction - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def metric_row(
    model: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected: bool,
) -> dict[str, object]:
    """生成 metrics.csv 中的一行；一行代表一个模型在一个数据部分上的整体表现。"""
    metrics = regression_metrics(y_true, y_pred)
    return {
        "model": model,
        "split": split,
        "n_samples": int(len(y_true)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "selected": bool(selected),
    }


def prediction_frame(
    model: str,
    split: str,
    target_index: np.ndarray,
    target_time: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """生成 predictions.csv 中的一组逐条预测记录。"""
    count = len(y_true)
    if not all(len(item) == count for item in (target_index, target_time, y_pred)):
        raise ValueError("target_index、target_time、y_true、y_pred 必须逐行一一对应")
    return pd.DataFrame(
        {
            "model": model,
            "split": split,
            "target_index": np.asarray(target_index, dtype=int),
            "target_time": pd.to_datetime(target_time),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def verify_prediction_table(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """从保存的逐条预测重新计算指标，核对它与 metrics.csv 一致。"""
    duplicated = predictions.duplicated(subset=["model", "split", "target_index"])
    if duplicated.any():
        raise ValueError(f"predictions.csv 有 {int(duplicated.sum())} 条重复的模型/数据部分/目标记录")

    recalculated = (
        predictions.groupby(["model", "split"], sort=False)
        .apply(lambda group: pd.Series(regression_metrics(group["y_true"], group["y_pred"])), include_groups=False)
        .reset_index()
    )
    merged = metrics.merge(recalculated, on=["model", "split"], suffixes=("_saved", "_again"))
    if len(merged) != len(metrics):
        raise ValueError("汇总指标和逐条预测记录的模型/数据部分组合不一致")
    for name in ("mae", "rmse"):
        if not np.allclose(merged[f"{name}_saved"], merged[f"{name}_again"], atol=1e-10):
            raise ValueError(f"保存后的 {name} 与从 predictions.csv 重算的结果不一致")
