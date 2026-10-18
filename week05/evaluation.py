"""第五周 · 评价组件（evaluation.py）

职责：只做"预测 + 算指标"。**不反传、不更新参数、不训练。**

本文件刻意不导入 training.py，保证"评价"永远碰不到训练逻辑
（training.py 反过来导入本文件，依赖方向是单向的，不会循环导入）。

三个容易混淆的尺度，这里分得很清楚：
1. 标准化目标尺度：训练过程中每轮记录 MSE 用这个尺度（数值小、好比较）。
2. 原始目标尺度：最后汇报 MAE / RMSE 用这个尺度（单位跟病情指标一致，人能看懂）。
3. 两者换算：pred_raw = pred_scaled * y_std + y_mean。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def predict_tensor(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """对给定输入做一次纯预测，返回 numpy 数组，形状 (N, 1)。

    两个关键点（组会会被问到）：
    - model.eval()：切换到评价模式。本周网络里只有 Linear 和 ReLU，
      没有 Dropout / BatchNorm，所以数值上没区别；但这是必须养成的习惯，
      一旦以后加了 Dropout，忘了切就会让评价结果不稳定。
    - torch.no_grad()：关闭自动求导，不建计算图、不存中间量。
      评价只关心结果，不需要梯度，关掉能省内存也更快。
    """
    model.eval()
    with torch.no_grad():
        output = model(x)
    return output.detach().cpu().numpy()


def mse_scaled(model: nn.Module, x: torch.Tensor, y_scaled: torch.Tensor) -> float:
    """在**标准化目标**尺度上算整份数据的 MSE。

    用途：训练循环里每轮记录训练集 / 验证集的曲线（也就是 history.csv 的两列 MSE）。
    注意它评价的是"传给它的那份完整数据"，不是某个批次。
    """
    prediction = predict_tensor(model, x)
    target = y_scaled.detach().cpu().numpy()
    return float(np.mean((prediction - target) ** 2))


def restore_scale(prediction_scaled: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    """把标准化尺度上的预测还原成原始目标单位。

    pred_raw = pred_scaled * y_std + y_mean
    这是训练时 (y - y_mean) / y_std 的逆运算。
    """
    return prediction_scaled * y_std + y_mean


def regression_metrics(y_true: np.ndarray, y_prediction: np.ndarray) -> dict[str, float]:
    """在原始目标单位上算 MAE、MSE、RMSE。

    - MAE：平均绝对误差，单位和目标一样，"平均差多少"，最直观。
    - MSE：均方误差，单位是目标的平方，对大误差惩罚更重。
    - RMSE：均方根误差，把 MSE 开根号，单位回到目标本身，和 MAE 可横向比较。
    三者都要求 y_true 与 y_prediction 的样本行一一对应。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prediction = np.asarray(y_prediction, dtype=float).reshape(-1)
    if y_true.shape != y_prediction.shape:
        raise ValueError(
            f"真实值与预测值形状不一致：{y_true.shape} vs {y_prediction.shape}"
        )
    error = y_prediction - y_true
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def evaluate_mlp_raw(
    model: nn.Module,
    x: torch.Tensor,
    y_raw: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, float]:
    """对已恢复最佳参数的 MLP，在原始单位上给出 MAE / RMSE。

    流程：模型输出标准化预测 → 用训练集的 y_mean / y_std 还原 → 和原始目标比较。
    """
    prediction_scaled = predict_tensor(model, x)
    prediction_raw = restore_scale(prediction_scaled, y_mean, y_std)
    return regression_metrics(y_raw, prediction_raw)


def evaluate_mlp_scaled(
    model: nn.Module,
    x: torch.Tensor,
    y_scaled: torch.Tensor,
) -> dict[str, float]:
    """在标准化尺度上给出 MSE，用于"从最佳轮次恢复参数后再复核一次曲线数值"。"""
    return {"MSE_scaled": mse_scaled(model, x, y_scaled)}
