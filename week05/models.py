"""第五周 · 模型组件（models.py）

职责：只说"模型长什么样"。不加载数据、不训练、不画图。

包含：
1. 实践一用的单层线性模型（权重和偏置都初始化为 0，方便手算核对）。
2. 实践二用的 MLP（一个隐藏层 + ReLU + 线性输出），隐藏层宽度可配。
3. 一个前向形状跟踪函数，用来把"形状怎么变"打印出来。

MLP 的单条样本公式（论文和课件里常见写法）：
    隐藏层：h = ReLU(W1 × x + b1)      x 是 10 维输入，h 是 hidden_dim 维隐藏特征
    输出层：pred = W2 × h + b2         pred 是 1 个标准化目标值

最后不接 ReLU 的原因：目标做了标准化，真实值有正有负（例如 -1.8 这种），
如果最后加 ReLU，负值会被一律截成 0，模型就永远预测不出偏低的病情，表达能力被白白砍掉。
"""

from __future__ import annotations

import torch
from torch import nn

# ------------------------------ 实验设置 ------------------------------
N_FEATURES = 10          # diabetes 的 10 个输入特征
DEFAULT_HIDDEN_DIM = 16  # 隐藏层默认宽度（正式实验用 16；修改练习用 8）
# ---------------------------------------------------------------------


# =====================================================================
# 实践一：单层线性模型
# =====================================================================
def build_zeroed_linear() -> nn.Linear:
    """创建 nn.Linear(1, 1)，并把权重和偏置都置为 0。

    为什么置零：如果随机初始化，第一次更新的数值就依赖随机值，不便手算核对。
    置零后预测恒为 0，初始 MSE 可以手算（见 run_step.py 的验证输出）。
    只有这个小例子这样初始化；实践二的 MLP 用正常的随机初始化。
    """
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


# =====================================================================
# 实践二：多层感知机
# =====================================================================
def build_mlp(
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    input_dim: int = N_FEATURES,
) -> nn.Sequential:
    """创建一个单隐藏层的 MLP：Linear → ReLU → Linear。

    hidden_dim 同时决定两个位置（这是"隐藏层宽度修改"练习的关键）：
      - 第一层 Linear 的**输出**维数：Linear(input_dim, hidden_dim)
      - 第二层 Linear 的**输入**维数：Linear(hidden_dim, 1)
    所以只改第一层、不改第二层，形状就对不上；宽度必须一次改两处。

    注意：本函数只负责"造模型"。要固定初始参数，请先在调用方设置
    torch.manual_seed(42)，这样两次调用会得到完全相同的初始权重。
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),   # 10 个输入特征 → hidden_dim 个隐藏特征
        nn.ReLU(),                          # 非线性：把负数截成 0，正数保留
        nn.Linear(hidden_dim, 1),           # hidden_dim 个隐藏特征 → 1 个预测值
    )


def forward_shape_trace(model: nn.Module, x: torch.Tensor) -> list[tuple[int, ...]]:
    """逐层跑一次前向，返回每一步的形状，用来验证"形状怎么变"。

    返回示例（32 条样本、hidden_dim=16）：
        [(32, 10), (32, 16), (32, 16), (32, 1)]
          输入      第一层Linear  ReLU后不变   输出层为1个预测值

    实现说明：不使用 model(x) 一步到底，而是手动逐层调用，
    这样才能把中间形状抓出来打印；整个检查包在 no_grad 里，不产生梯度。
    """
    shapes: list[tuple[int, ...]] = [tuple(x.shape)]
    current = x
    with torch.no_grad():
        for layer in model:
            current = layer(current)
            shapes.append(tuple(current.shape))
    return shapes


def parameter_count(model: nn.Module) -> dict[str, int]:
    """数一数每层有多少个可学习参数，用于说明"隐藏层宽度改了多少参数"。

    Linear 的参数个数 = 输入维数 × 输出维数 + 输出维数（权重 + 偏置）。
    """
    counts: dict[str, int] = {}
    for index, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weight = layer.weight.numel()
            bias = layer.bias.numel() if layer.bias is not None else 0
            counts[f"layer{index}_Linear({layer.in_features}->{layer.out_features})"] = weight + bias
    counts["total"] = sum(counts.values())
    return counts
