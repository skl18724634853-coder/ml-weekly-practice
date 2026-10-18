"""第五周 · 数据组件（data.py）

职责：只负责"把数据变成模型能吃的东西"，不涉及任何模型与训练逻辑。

包含两件事：
1. 实践一：两个样本的极小张量（用于对照一次梯度更新）。
2. 实践二：diabetes 的划分、标准化、张量与批次准备。

关于划分与标准化（与第三、四周保持同一套规范）：
- 先按 60% / 20% / 20% 划分训练、验证、测试，固定随机种子 42，所有模型共用同一份划分。
- StandardScaler 只在训练集上 fit，验证、测试集只用训练集得到的统计量 transform。
- 目标也做标准化：(y - y_mean) / y_std，其中 y_mean、y_std 同样只来自训练集。
  保留原始目标与这两个统计量，评价时用 pred_raw = pred_scaled * y_std + y_mean 恢复尺度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------ 实验设置 ------------------------------
SEED = 42                 # 划分、初始化、打乱顺序共用的随机种子
BATCH_SIZE = 32           # 小批量大小（实践二）
TRAIN_RATIO = 0.6         # 约 60% 训练
VALIDATION_RATIO = 0.2    # 约 20% 验证
TEST_RATIO = 0.2          # 约 20% 测试
# ---------------------------------------------------------------------


# =====================================================================
# 实践一：两个样本
# =====================================================================
def make_step_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """实践一的样本：x = [[1.0], [2.0]]，y = [[2.0], [4.0]]。

    形状都是 (2, 1)，表示"2 个样本、每个样本 1 个特征 / 1 个目标值"。
    转成 float32 是因为 PyTorch 的默认计算类型就是 float32。
    """
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    return x, y


# =====================================================================
# 实践二：diabetes 的划分、标准化、张量
# =====================================================================
@dataclass
class DiabetesData:
    """一次性装好实践二需要用的全部数据，避免在各个组件之间反复重算。

    字段说明（N 为样本数，10 为特征数）：
    - x_train_scaled / x_validation_scaled / x_test_scaled：numpy 数组，形状 (N, 10)
        给基线模型（均值基线、线性回归）用，它们是 sklearn 的实现，吃 numpy。
    - x_train_tensor / x_validation_tensor / x_test_tensor：torch.float32，(N, 10)
        给神经网络用；每轮整份评价时直接喂这几个张量。
    - y_train_scaled / y_validation_scaled / y_test_scaled：torch.float32，(N, 1)
        标准化后的目标，训练和算"标准化单位 MSE"时用。
    - y_train_raw / y_validation_raw / y_test_raw：numpy 数组，(N,)
        原始目标，最后算 MAE / RMSE 时用。
    - y_mean / y_std：训练目标的均值与标准差，用来在标准化尺度和原始尺度之间换算。
    - feature_names：10 个特征的名字，便于打印与核对。
    """

    x_train_scaled: np.ndarray
    x_validation_scaled: np.ndarray
    x_test_scaled: np.ndarray
    x_train_tensor: torch.Tensor
    x_validation_tensor: torch.Tensor
    x_test_tensor: torch.Tensor
    y_train_scaled: torch.Tensor
    y_validation_scaled: torch.Tensor
    y_test_scaled: torch.Tensor
    y_train_raw: np.ndarray
    y_validation_raw: np.ndarray
    y_test_raw: np.ndarray
    y_mean: float
    y_std: float
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def split_indices(sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """固定随机种子，先留出 40%，再平分得到约 60% / 20% / 20% 的三份行下标。

    所有模型共用这一份划分，比较才公平。
    """
    indices = np.arange(sample_count)
    train_idx, remaining_idx = train_test_split(
        indices, test_size=VALIDATION_RATIO + TEST_RATIO, random_state=SEED
    )
    validation_idx, test_idx = train_test_split(
        remaining_idx, test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
        random_state=SEED,
    )
    return train_idx, validation_idx, test_idx


def prepare_diabetes() -> DiabetesData:
    """读取 diabetes 数据，完成划分、标准化、转张量。

    这里刻意 keep 原始目标（原始尺度），因为最后交的 MAE / RMSE 要求是原始单位。
    """
    dataset = load_diabetes(as_frame=True, scaled=False)  # 用原始数值，不用 sklearn 预设的标准化版本
    x_frame = dataset.data
    y_array = dataset.target.to_numpy(dtype=float)

    train_idx, validation_idx, test_idx = split_indices(len(x_frame))
    x_train = x_frame.iloc[train_idx]
    x_validation = x_frame.iloc[validation_idx]
    x_test = x_frame.iloc[test_idx]
    y_train_raw = y_array[train_idx]
    y_validation_raw = y_array[validation_idx]
    y_test_raw = y_array[test_idx]

    # 输入标准化：只在训练集上 fit，验证/测试沿用训练集的均值和标准差
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    # 目标标准化：同样只用训练集统计量；std 用 ddof=0（总体标准差）
    y_mean = float(y_train_raw.mean())
    y_std = float(np.std(y_train_raw, ddof=0))
    if not y_std > 0:
        raise ValueError(f"训练目标的标准差必须大于 0，实际为 {y_std}")
    y_train_scaled = (y_train_raw - y_mean) / y_std
    y_validation_scaled = (y_validation_raw - y_mean) / y_std
    y_test_scaled = (y_test_raw - y_mean) / y_std

    def as_float_tensor(values: np.ndarray, columns: int) -> torch.Tensor:
        """统一转成 float32 张量；目标补一列变成 (N, 1)，和模型输出形状对齐。"""
        tensor = torch.tensor(values, dtype=torch.float32)
        if columns == 1 and tensor.dim() == 1:
            tensor = tensor.view(-1, 1)
        return tensor

    return DiabetesData(
        x_train_scaled=x_train_scaled,
        x_validation_scaled=x_validation_scaled,
        x_test_scaled=x_test_scaled,
        x_train_tensor=as_float_tensor(x_train_scaled, x_train_scaled.shape[1]),
        x_validation_tensor=as_float_tensor(x_validation_scaled, x_validation_scaled.shape[1]),
        x_test_tensor=as_float_tensor(x_test_scaled, x_test_scaled.shape[1]),
        y_train_scaled=as_float_tensor(y_train_scaled, 1),
        y_validation_scaled=as_float_tensor(y_validation_scaled, 1),
        y_test_scaled=as_float_tensor(y_test_scaled, 1),
        y_train_raw=y_train_raw,
        y_validation_raw=y_validation_raw,
        y_test_raw=y_test_raw,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=x_frame.columns.tolist(),
    )


def make_train_loader(
    data: DiabetesData, batch_size: int = BATCH_SIZE, shuffle_seed: int = SEED
) -> DataLoader:
    """为一次训练新建 DataLoader：把特征和标签配对，并按 batch_size 分批。

    - TensorDataset 保证"第 i 行的特征"和"第 i 行的标签"始终配对，不会错位。
    - drop_last=False：最后一批不足 batch_size 也参与训练（不丢掉样本）。
    - 每个 DataLoader 用自己新建的 Generator，保证两组配置每轮的样本顺序一致，
      这样 SGD 与 Adam 的差别只来自优化器，而不是样本顺序。
    """
    dataset = TensorDataset(data.x_train_tensor, data.y_train_scaled)
    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,   # Windows 上多进程容易出问题；数据很小，单进程足够
        generator=generator,
    )
