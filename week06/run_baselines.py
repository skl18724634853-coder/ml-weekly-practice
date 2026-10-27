"""实践二入口：ETTh1 的单步序列预测基线、Ridge 调参与最终评价。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import check_hourly_time_index, load_etth1, make_windows, select_rows, split_by_target_time
from evaluation import metric_row, prediction_frame, regression_metrics, verify_prediction_table
from models import fit_lag_linear, fit_ridge, predict_last_value, predict_window_mean
from reporting import plot_series_split, plot_tuning_comparison, plot_validation_predictions, save_csv


RESULTS_DIR = Path(__file__).parent / "results"
SERIES_LENGTH = 14_400
WINDOW_SIZE = 24
TRAIN_END = pd.Timestamp("2017-06-26 00:00:00")
VALIDATION_END = pd.Timestamp("2017-10-24 00:00:00")
TEST_END = pd.Timestamp("2018-02-21 00:00:00")
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
MODEL_ORDER = ("last_value", "window_mean", "lag_linear", "ridge_tuned")


def use_utf8_output() -> None:
    """避免 Windows 终端中的中文输出乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def subset_dict(
    x: np.ndarray,
    y: np.ndarray,
    target_index: np.ndarray,
    target_time: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """同一个时间掩码同时作用于 X、y、下标和时间，防止预测与标签错位。"""
    return {
        split: {
            "x": select_rows(x, mask),
            "y": select_rows(y, mask),
            "target_index": select_rows(target_index, mask),
            "target_time": select_rows(target_time, mask),
        }
        for split, mask in masks.items()
    }


def evaluate_predictor(
    model_name: str,
    predictor,
    splits: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], list[pd.DataFrame], dict[str, np.ndarray]]:
    """对一个预测方法统一产生三份指标、三份逐条预测和预测数组。"""
    metric_rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    predictions: dict[str, np.ndarray] = {}
    for split, part in splits.items():
        pred = np.asarray(predictor(part["x"]), dtype=float).reshape(-1)
        predictions[split] = pred
        metric_rows.append(metric_row(model_name, split, part["y"], pred, selected=False))
        prediction_tables.append(
            prediction_frame(
                model_name,
                split,
                part["target_index"],
                part["target_time"],
                part["y"],
                pred,
            )
        )
    return metric_rows, prediction_tables, predictions


def tuning_row(model: str, param_name: str, param_value: float | None, train_y: np.ndarray, train_pred: np.ndarray, val_y: np.ndarray, val_pred: np.ndarray) -> dict[str, object]:
    """生成 tuning.csv 的一行；训练和验证误差都在原始油温尺度计算。"""
    train = regression_metrics(train_y, train_pred)
    validation = regression_metrics(val_y, val_pred)
    return {
        "model": model,
        "param_name": param_name,
        "param_value": param_value,
        "train_mae": train["mae"],
        "val_mae": validation["mae"],
        "train_rmse": train["rmse"],
        "val_rmse": validation["rmse"],
        "rmse_gap": validation["rmse"] - train["rmse"],
        "chosen_candidate": False,
    }


def main() -> None:
    use_utf8_output()
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1) 读取前 14,400 小时原始记录，并检查时间与数值。
    source = load_etth1()
    experiment = source.iloc[:SERIES_LENGTH].copy()
    check_hourly_time_index(experiment["date"])
    if len(experiment) != SERIES_LENGTH:
        raise ValueError(f"需要前 {SERIES_LENGTH} 小时，实际读取到 {len(experiment)} 行")
    series = experiment["OT"].to_numpy(dtype=float)
    times = experiment["date"].to_numpy()
    print(f"ETTh1 实验记录：{experiment['date'].iloc[0]} 到 {experiment['date'].iloc[-1]}，共 {len(experiment)} 小时")

    # 2) 先构造窗口，再按目标时刻划分。首条验证/测试样本可使用边界前的真实历史。
    x, y, target_index = make_windows(series, WINDOW_SIZE)
    target_time = times[target_index]
    masks = split_by_target_time(target_time, TRAIN_END, VALIDATION_END, TEST_END)
    splits = subset_dict(x, y, target_index, target_time, masks)
    print(f"窗口形状：X={x.shape}，y={y.shape}；窗口长度 L={WINDOW_SIZE}")
    for split, part in splits.items():
        first_time = pd.Timestamp(part["target_time"][0])
        last_time = pd.Timestamp(part["target_time"][-1])
        print(f"  {split:10s}：{len(part['y']):5d} 条目标，{first_time} 到 {last_time}")
    for split in ("validation", "test"):
        first_target = int(splits[split]["target_index"][0])
        print(
            f"首条 {split}：输入 {pd.Timestamp(times[first_target - WINDOW_SIZE])} 到 "
            f"{pd.Timestamp(times[first_target - 1])}，目标 {pd.Timestamp(times[first_target])}"
        )

    # 3) 三种基线：两种固定规则不拟合；lag_linear 只在训练窗口上拟合。
    lag_linear = fit_lag_linear(splits["train"]["x"], splits["train"]["y"])
    lag_train = lag_linear.predict(splits["train"]["x"])
    lag_validation = lag_linear.predict(splits["validation"]["x"])
    tuning_rows: list[dict[str, object]] = [
        tuning_row(
            "lag_linear", "", None,
            splits["train"]["y"], lag_train,
            splits["validation"]["y"], lag_validation,
        )
    ]

    # 4) Ridge 调参：同一份训练窗口拟合三个 alpha，验证 RMSE 最小者胜出；平局取较小 alpha。
    ridge_candidates: list[tuple[float, object, dict[str, object]]] = []
    for alpha in RIDGE_ALPHAS:
        model = fit_ridge(splits["train"]["x"], splits["train"]["y"], alpha)
        row = tuning_row(
            "ridge", "alpha", alpha,
            splits["train"]["y"], model.predict(splits["train"]["x"]),
            splits["validation"]["y"], model.predict(splits["validation"]["x"]),
        )
        ridge_candidates.append((alpha, model, row))
        tuning_rows.append(row)

    chosen_alpha, tuned_ridge, chosen_row = min(ridge_candidates, key=lambda item: (item[2]["val_rmse"], item[0]))
    chosen_row["chosen_candidate"] = True
    tuning = pd.DataFrame(tuning_rows)
    save_csv(tuning, RESULTS_DIR / "tuning.csv")
    plot_tuning_comparison(tuning, RESULTS_DIR / "tuning_comparison.png")
    print(f"Ridge 调参完成：选定 alpha={chosen_alpha:g}，验证 RMSE={chosen_row['val_rmse']:.4f}")

    # 5) 固定配置后，比较四个最终方法。测试集只在此处参与一次最终评价。
    predictors = {
        "last_value": predict_last_value,
        "window_mean": predict_window_mean,
        "lag_linear": lag_linear.predict,
        "ridge_tuned": tuned_ridge.predict,
    }
    all_metric_rows: list[dict[str, object]] = []
    all_prediction_tables: list[pd.DataFrame] = []
    validation_rmse: dict[str, float] = {}
    for name in MODEL_ORDER:
        rows, tables, _ = evaluate_predictor(name, predictors[name], splits)
        all_metric_rows.extend(rows)
        all_prediction_tables.extend(tables)
        validation_rmse[name] = next(row["rmse"] for row in rows if row["split"] == "validation")

    selected_model = min(MODEL_ORDER, key=lambda name: (validation_rmse[name], MODEL_ORDER.index(name)))
    metrics = pd.DataFrame(all_metric_rows)
    metrics["selected"] = metrics["model"].eq(selected_model)
    predictions = pd.concat(all_prediction_tables, ignore_index=True).sort_values(
        ["model", "split", "target_index"]
    )
    verify_prediction_table(predictions, metrics)
    save_csv(metrics, RESULTS_DIR / "metrics.csv")
    save_csv(predictions, RESULTS_DIR / "predictions.csv")

    # 6) 两张场景图：完整时段划分，及验证首周的预测/负载对照。
    plot_series_split(
        experiment,
        TRAIN_END,
        VALIDATION_END,
        experiment["date"].iloc[WINDOW_SIZE - 1],
        RESULTS_DIR / "series_split.png",
    )
    plot_validation_predictions(predictions, experiment, TRAIN_END, RESULTS_DIR / "validation_predictions.png")

    print("\n最终验证 RMSE：")
    for name in MODEL_ORDER:
        print(f"  {name:12s}: {validation_rmse[name]:.4f}{'  <- selected' if name == selected_model else ''}")
    print("\n最终测试指标：")
    print(metrics[metrics["split"] == "test"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n已生成所有结果到：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
