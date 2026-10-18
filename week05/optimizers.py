"""第五周 · 优化器组件（optimizers.py）

职责：只负责"根据名字和超参数造出优化器"。不训练、不评价、不碰数据。

为什么要单独一个文件：
组会上要能"指着代码说清楚 SGD 和 Adam 用的是哪两套配置"。把它们集中在这里，
两套配置一目了然，也避免参数散落在训练逻辑里。

优化器在训练里的角色：
- 它自己不算梯度（梯度由 loss.backward() 自动求导得到，存在 param.grad 里）。
- 它只负责在 optimizer.step() 时读取 param.grad，按自己的规则修改 param.data。
"""

from __future__ import annotations

import torch
from torch import nn

# ------------------------------ 实验设置 ------------------------------
# 实践二的两套配置。两组都从"从未训练"的相同初始参数出发，
# 差别只有优化器算法和它自己的学习率。
MLP_OPTIMIZER_CONFIGS: tuple[dict[str, float | str], ...] = (
    {
        "name": "sgd",
        "label": "MLP + SGD",
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
    },
    {
        "name": "adam",
        "label": "MLP + Adam",
        "learning_rate": 0.001,
        "momentum": 0.0,   # Adam 不用 momentum 这个参数，其余保持默认
        "weight_decay": 0.0,
    },
)

# 实践一：学习率 0.1、不带动量和权重衰减的 SGD
STEP_LEARNING_RATE = 0.1
# ---------------------------------------------------------------------


def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """按名字创建优化器。

    参数
    ----
    optimizer_name : "sgd" 或 "adam"
    model          : 优化器要管理的模型；model.parameters() 会把它所有可学习参数交出来
    learning_rate  : 学习率，两套配置分别是 0.01（SGD）和 0.001（Adam）
    momentum       : 仅 SGD 使用；Adam 保持默认不用这个参数
    weight_decay   : L2 惩罚系数，本周两组都设为 0

    两种更新规则的直觉（不要求背公式）：
    - SGD：只看"当前这一步的梯度"，直接按 参数 -= 学习率 × 梯度 走一步。
      好处是简单直接；坏处是用法固定、方向容易被单个批次的噪声带偏。
    - Adam：除了当前梯度，还记住"历史梯度的平均方向"和"历史梯度的波动大小"，
      用两者对每个参数单独缩放步长。所以它更稳、对学习率不那么敏感，
      但**仍然需要学习率**——学习率决定"整体步子迈多大"，
      Adam 只调整"各个参数之间步子怎么分配"。
    """
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        # 其余参数（betas、eps）全部保持 PyTorch 默认值
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"未知的优化器名称：{optimizer_name!r}，只支持 'sgd' 或 'adam'")


def config_by_name(name: str) -> dict[str, float | str]:
    """按名字取出 MLP_OPTIMIZER_CONFIGS 里的那一条配置，供入口统一遍历。"""
    for config in MLP_OPTIMIZER_CONFIGS:
        if config["name"] == name:
            return dict(config)
    raise ValueError(f"未知的配置名：{name!r}")
