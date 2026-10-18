"""第五周 · 训练组件（training.py）

职责：**最核心的一块**。存放"一次参数更新""一轮训练""完整训练"以及基线的拟合。

组会上要能指着本文件说清楚两件事：
1. 四个核心步骤分别在哪一行（清梯度 / 前向 / 损失 / 反传 / 更新）。
2. 为什么只算最终输出的损失，前面第一层也能学到东西
   —— 因为 loss.backward() 会沿计算图自动把梯度一路乘回每一层（链式法则）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.utils.data import DataLoader

import evaluation


# =====================================================================
# 实践一：一次更新，并记录每个阶段的参数与梯度
# =====================================================================
@dataclass
class UpdateTrace:
    """一次更新的完整记录，供入口核对与写 CSV。

    rows：三行，分别对应 清梯度后 / 反传后 / 更新后。
    initial_prediction / updated_prediction：更新前后模型对同两个样本的预测。
        这两个值用来演示"更新后重新前向才会变，旧预测不会自动重算"。
    loss_before / loss_after：更新前算出的损失、更新后重新前向再算的损失。
    """

    rows: list[dict[str, float | str | None]] = field(default_factory=list)
    initial_prediction: np.ndarray | None = None
    updated_prediction: np.ndarray | None = None
    loss_before: float | None = None
    loss_after: float | None = None


def _pairs(weight: torch.Tensor, bias: torch.Tensor) -> tuple[float, float]:
    """把权重、偏置取成 python 浮点数，方便写 CSV 和打印。"""
    return float(weight.item()), float(bias.item())


def one_update_trace(
    model: nn.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> UpdateTrace:
    """按"清梯度 → 前向 → 损失 → 反传 → 更新"走一次，并在三个阶段各记一笔。

    对应第二周手写的公式：
        dw = 平均(2 × (pred - y) × x)        ← 这一步现在由 loss.backward() 完成
        w  = w - 学习率 × dw                  ← 这一步现在由 optimizer.step() 完成
    """

    def record(stage: str, loss: float | None) -> dict[str, float | str | None]:
        """抓取当前参数与梯度。梯度还没算出来时是 None，CSV 里留空。"""
        weight, bias = _pairs(model.weight, model.bias)
        weight_grad = None if model.weight.grad is None else float(model.weight.grad.item())
        bias_grad = None if model.bias.grad is None else float(model.bias.grad.item())
        return {
            "stage": stage,
            "weight": weight,
            "bias": bias,
            "weight_grad": weight_grad,
            "bias_grad": bias_grad,
            "loss": loss,
        }

    trace = UpdateTrace()

    # ① 清梯度：把上一次（这里是第一次）残留的 grad 清掉，否则梯度会累加
    optimizer.zero_grad()
    trace.rows.append(record("after_zero_grad", None))

    # ② 前向：用当前权重和偏置计算预测
    prediction = model(x)
    # 样本配对检查：预测和标签必须都是 (2, 1)，行数、列数都要一致
    assert prediction.shape == y.shape, f"预测 {prediction.shape} 与标签 {y.shape} 不配对"

    # ③ 损失：把预测和对应标签比较，得到 MSE 标量
    loss = loss_fn(prediction, y)
    trace.loss_before = float(loss.item())

    # ④ 反传：自动求导把 loss 对每个参数的梯度算出来，写进 param.grad
    #    这一步等价于第二周手写的 dw、db —— 只是现在不用自己推公式、也不用自己乘链式法则
    loss.backward()
    trace.rows.append(record("after_backward", trace.loss_before))

    # ⑤ 更新：优化器读取 param.grad，按自己的规则修改 param.data
    #    注意：step() 只改参数，不改梯度；梯度仍然是刚才算出来的那份
    optimizer.step()
    weight_after, bias_after = _pairs(model.weight, model.bias)

    # 更新后必须**重新前向**才能得到新预测；上面保存的 prediction 是旧值的快照，不会自动重算
    with torch.no_grad():
        updated_prediction = model(x)
        loss_after = float(loss_fn(updated_prediction, y).item())

    trace.rows.append(record("after_step", loss_after))
    trace.initial_prediction = prediction.detach().numpy().copy()
    trace.updated_prediction = updated_prediction.detach().numpy().copy()
    trace.loss_after = loss_after

    # 一个额外核对：新参数必须等于"旧参数 - 学习率 × 梯度"
    learning_rate = float(optimizer.param_groups[0]["lr"])
    trace_rows = trace.rows
    assert abs(weight_after - (trace_rows[1]["weight"] - learning_rate * trace_rows[1]["weight_grad"])) < 1e-6
    assert abs(bias_after - (trace_rows[1]["bias"] - learning_rate * trace_rows[1]["bias_grad"])) < 1e-6

    return trace


# =====================================================================
# 实践二：一批 → 一轮 → 完整训练
# =====================================================================
def train_one_batch(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """对**一个批次**做一次完整更新，返回该批次的损失。

    xb、yb 是当前批次的数据（不是整个训练集）——这一点组会会被问到（A5）：
    本次算出的梯度只用到了这一批的样本，下一批会换成另一批样本，梯度自然不同。
    """
    optimizer.zero_grad()          # ① 清掉上一批留下的梯度
    prediction = model(xb)         # ② 前向：本批样本过网络
    loss = loss_fn(prediction, yb)  # ③ 损失：与本批标签比较
    loss.backward()                # ④ 反传：自动求梯度（只涉及本批样本）
    optimizer.step()               # ⑤ 更新：按梯度改参数
    return float(loss.item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """跑完一轮（遍历训练集里的所有批次），返回本轮的批次损失统计。

    返回 last_batch_loss 是为了说明组会问题 B4：
    最后一批的 loss **不能**直接当作"这一轮结束时的训练 MSE"，原因有两个
    —— 它只覆盖最后一批样本（不足 batch_size 的那批可能只有 9 条），
    并且它是用"这一批更新之前"的参数算出来的。
    """
    model.train()                  # 每轮开始前切回训练模式
    batch_losses: list[float] = []
    for xb, yb in loader:
        batch_losses.append(train_one_batch(model, xb, yb, loss_fn, optimizer))
    return {
        "mean_batch_loss": float(np.mean(batch_losses)),
        "last_batch_loss": float(batch_losses[-1]),
        "batch_count": float(len(batch_losses)),
    }


@dataclass
class BestModel:
    """记录验证集表现最好的那一轮，以及那一轮对应的模型参数。"""

    epoch: int | None = None
    validation_mse: float = float("inf")
    state_dict: dict[str, torch.Tensor] | None = None


def fit_mlp(
    model: nn.Module,
    loader: DataLoader,
    data,
    epochs: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    label: str,
    learning_rate: float,
) -> tuple[list[dict[str, object]], BestModel]:
    """完整训练：跑 epochs 轮，每轮结束后评价完整训练集与完整验证集。

    每轮记录（对应 history.csv 的一行）：
      - train_MSE / validation_MSE：**用整份数据重新评价**得到，标准化目标尺度
      - last_batch_loss：最后一批的批次损失，用来对比说明"它不等于整轮训练 MSE"
      - is_best：这一轮是否是验证集最优轮次（循环结束后回填）

    为什么不能直接把最后一批 loss 当作这一轮训练 MSE（组会 B4）：
    最后一批只有 9 条样本、且是用更新前的参数算的，而整份训练集是 265 条、
    用的是这一轮所有更新之后的参数。
    """
    history: list[dict[str, object]] = []
    best = BestModel()

    for epoch in range(1, epochs + 1):
        batch_stats = train_one_epoch(model, loader, loss_fn, optimizer)

        # 每轮更新结束后，用**当时的同一个模型**重新评价完整的训练集和验证集
        train_mse = evaluation.mse_scaled(model, data.x_train_tensor, data.y_train_scaled)
        validation_mse = evaluation.mse_scaled(
            model, data.x_validation_tensor, data.y_validation_scaled
        )

        history.append({
            "config": label,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "train_MSE": train_mse,
            "validation_MSE": validation_mse,
            "last_batch_loss": batch_stats["last_batch_loss"],
            "target_scale": "standardized",
            "is_best": False,
        })

        # 只有"严格优于历史最佳"才替换，这样验证 MSE 相同时会保留更早的一轮
        if validation_mse < best.validation_mse:
            best.epoch = epoch
            best.validation_mse = validation_mse
            best.state_dict = copy.deepcopy(model.state_dict())

    if best.epoch is not None:
        history[best.epoch - 1]["is_best"] = True
    return history, best


def load_best_state(model: nn.Module, best: BestModel) -> None:
    """把模型参数恢复成"最佳轮次"的那一份。

    注意这不是"继续训练"，只是把权重换成当时保存的快照，
    所以恢复之后模型反映的是最佳轮次的状态，而不是最后一轮的状态（组会 B5）。
    """
    if best.state_dict is None:
        raise RuntimeError("没有记录到最佳轮次参数")
    model.load_state_dict(best.state_dict)


# =====================================================================
# 基线模型
# =====================================================================
def fit_mean_baseline(y_train_raw: np.ndarray) -> float:
    """均值基线：始终用训练目标的平均值作为预测。

    它不学习任何特征，只给其他模型立一个"什么都不做"的参照线。
    如果某个模型连它都比不过，说明这个模型没学到东西。
    """
    return float(np.mean(y_train_raw))


def fit_linear_baseline(x_train_scaled: np.ndarray, y_train_raw: np.ndarray) -> LinearRegression:
    """线性回归基线：标准化输入 + 原始目标。

    用原始目标而不是标准化目标，是为了让它的预测直接落在原始单位上，
    与 MLP 恢复尺度后的预测、以及 MAE / RMSE 的口径完全一致。
    """
    model = LinearRegression()
    model.fit(x_train_scaled, y_train_raw)
    return model
