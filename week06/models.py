"""第六周 · 三种预测基线与 Ridge 调参路线。"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_last_value(x: np.ndarray) -> np.ndarray:
    """朴素预测：下一小时等于窗口最后一列，也就是最近一次真实观测。"""
    values = np.asarray(x, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"窗口 X 应是二维数组，实际形状为 {values.shape}")
    return values[:, -1]


def predict_window_mean(x: np.ndarray) -> np.ndarray:
    """窗口均值预测：下一小时等于窗口中所有历史油温的平均。"""
    values = np.asarray(x, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"窗口 X 应是二维数组，实际形状为 {values.shape}")
    return values.mean(axis=1)


def fit_lag_linear(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """拟合“标准化 24 个滞后特征 -> 线性回归”的基准模型。"""
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    return model


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Pipeline:
    """拟合 Ridge 路线；alpha 越大，L2 正则化对权重的约束越强。"""
    if alpha <= 0:
        raise ValueError(f"Ridge 的 alpha 必须为正数，实际为 {alpha}")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return model
