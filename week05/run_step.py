"""实践一入口（run_step.py）：把一次梯度更新与代码对应起来

这一份是"最小可核对"的例子，目的是把第二周手推的梯度下降和 PyTorch 对上：
    dw、db 的手写计算  →  loss.backward()
    手写参数更新        →  optimizer.step()

用两个样本 x=[[1],[2]]、y=[[2],[4]]，单层线性模型初始化为 0，
学习率 0.1 的 SGD。跑完会打印三行记录（清梯度后 / 反传后 / 更新后）并核对数值。

运行：进入 week05 后执行  python run_step.py
"""

from __future__ import annotations

import sys

import torch
from torch import nn

import data
import models
import optimizers
import reporting
import training


def use_utf8_output() -> None:
    """避免 Windows 终端输出中文时乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def format_prediction(array) -> str:
    """把 (2,1) 的预测数组压成一行 [a, b]，便于打印对照。"""
    return "[" + ", ".join(f"{float(value):.1f}" for value in array.reshape(-1)) + "]"


def print_trace_table(rows: list[dict]) -> None:
    """把三行记录打印成一张对齐的表，方便对着屏幕逐列讲解。"""
    print(f"{'阶段 stage':<18}{'weight':>10}{'bias':>10}{'weight_grad':>14}{'bias_grad':>12}{'loss':>10}")
    print("-" * 76)
    for row in rows:
        def fmt(value: object, digits: int = 4) -> str:
            return "—" if value is None else f"{float(value):.{digits}f}"

        print(
            f"{str(row['stage']):<18}"
            f"{fmt(row['weight']):>10}"
            f"{fmt(row['bias']):>10}"
            f"{fmt(row['weight_grad']):>14}"
            f"{fmt(row['bias_grad']):>12}"
            f"{fmt(row['loss']):>10}"
        )
    print("（'—' 表示该阶段还没有这个量：清梯度后尚未反传，所以没有梯度也没有损失）")
    print()


def main() -> None:
    use_utf8_output()

    # ---------- 1. 准备数据、模型、优化器、损失函数 ----------
    x, y = data.make_step_tensors()
    print("=" * 76)
    print("实践一：一次梯度更新的完整记录")
    print("=" * 76)
    print(f"样本特征 x 形状 {tuple(x.shape)}，内容 {x.tolist()}")
    print(f"样本标签 y 形状 {tuple(y.shape)}，内容 {y.tolist()}")
    print("模型：nn.Linear(1, 1)，权重和偏置都初始化为 0（方便手算核对）")
    print(f"优化器：SGD，学习率 {optimizers.STEP_LEARNING_RATE}，momentum=0，weight_decay=0")
    print("损失函数：nn.MSELoss()（回归任务用均方误差）")
    print()

    model = models.build_zeroed_linear()
    optimizer = optimizers.build_optimizer(
        "sgd", model, learning_rate=optimizers.STEP_LEARNING_RATE,
        momentum=0.0, weight_decay=0.0,
    )
    loss_fn = nn.MSELoss()

    # ---------- 2. 手算基线：初始预测、初始 MSE ----------
    with torch.no_grad():
        initial_prediction = model(x)
        initial_mse = float(loss_fn(initial_prediction, y).item())
    print(f"初始状态：权重 = 0，偏置 = 0 ⇒ 两个预测都是 0 ⇒ 初始 MSE = {initial_mse:.4f}")
    print("  手算核对：MSE = ((0-2)² + (0-4)²) / 2 = (4 + 16) / 2 = 10 ✓")
    assert abs(initial_mse - 10.0) < 1e-6, f"初始 MSE 应为 10，实际 {initial_mse}"
    print()

    # ---------- 3. 走一次完整更新并记录三行 ----------
    trace = training.one_update_trace(model, x, y, loss_fn, optimizer)
    print("三阶段记录：")
    print_trace_table(trace.rows)

    # ---------- 4. 逐项核对 ----------
    after_backward = trace.rows[1]
    after_step = trace.rows[2]
    learning_rate = optimizers.STEP_LEARNING_RATE

    print("核对 1：反传得到的梯度")
    print(f"  weight_grad = {after_backward['weight_grad']:.4f}，bias_grad = {after_backward['bias_grad']:.4f}")
    print("  手算：dw = 平均(2×(pred-y)×x) = (2×(0-2)×1 + 2×(0-4)×2)/2 = -10")
    print("        db = 平均(2×(pred-y))   = (2×(0-2) + 2×(0-4))/2       = -6")
    print()

    print("核对 2：新参数 = 旧参数 - 学习率 × 梯度")
    print(f"  weight: {after_backward['weight']:.3f} - {learning_rate} × ({after_backward['weight_grad']:.3f})"
          f" = {after_step['weight']:.4f}")
    print(f"  bias  : {after_backward['bias']:.3f} - {learning_rate} × ({after_backward['bias_grad']:.3f})"
          f" = {after_step['bias']:.4f}")
    print()

    print("核对 3：更新后必须重新前向，旧预测不会自动重算")
    print(f"  更新前保存的预测（旧快照）：{format_prediction(trace.initial_prediction)}")
    print(f"  更新后重新前向的预测      ：{format_prediction(trace.updated_prediction)}")
    print(f"  损失：更新前 {trace.loss_before:.4f} → 更新后重新计算 {trace.loss_after:.4f}")
    print("  手算核对：w=1、b=0.6 ⇒ 预测 1×1+0.6=1.6 和 1×2+0.6=2.6")
    print("            MSE = ((1.6-2)² + (2.6-4)²)/2 = (0.16 + 1.96)/2 = 1.06 ✓")
    print()

    # ---------- 5. 保存记录 ----------
    csv_path = reporting.save_step_trace(trace.rows)
    print(f"记录已保存：{csv_path}")
    print()
    print("=" * 76)
    print("小结：哪一行代码做了什么")
    print("=" * 76)
    print("  optimizer.zero_grad()     清掉上一轮的梯度，否则新梯度和旧梯度会累加")
    print("  pred = model(x)           前向：用当前参数算预测")
    print("  loss = loss_fn(pred, y)   损失：把预测和标签比较")
    print("  loss.backward()           反传：自动求导，梯度写进 param.grad（= 第二周手写的 dw、db）")
    print("  optimizer.step()          更新：按梯度和优化器规则改 param.data（= 第二周手写的 w -= lr*dw）")
    print("=" * 76)


if __name__ == "__main__":
    main()
