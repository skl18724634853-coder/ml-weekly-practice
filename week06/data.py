"""第六周 · ETTh1 序列读取、窗口构造与时间划分。

本文件只负责把连续的油温记录变成监督学习样本：
    过去 L 小时油温 -> 紧接着下一小时油温。

每一行输入 X[i] 都按“从旧到新”排列；y[i] 是窗口后紧接着的目标；
target_index[i] 和 target_time[i] 用于核对这条预测真实对应哪一个时刻。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "ETTh1.csv"
REQUIRED_COLUMNS = ("date", "HUFL", "OT")


def load_etth1(path: Path = DATA_PATH) -> pd.DataFrame:
    """读取 ETTh1，并检查本周需要的时间、油温和负载列。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    frame = pd.read_csv(path, parse_dates=["date"])
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"ETTh1 缺少需要的列：{missing}")
    if frame["date"].isna().any() or frame["OT"].isna().any() or frame["HUFL"].isna().any():
        raise ValueError("date、OT、HUFL 不应包含缺失值")
    return frame


def check_hourly_time_index(times: pd.Series) -> None:
    """确认时间严格递增、不重复，且相邻记录正好相差一小时。"""
    if not times.is_monotonic_increasing:
        raise ValueError("时间列必须按从早到晚排列")
    if times.duplicated().any():
        raise ValueError("时间列不能有重复时刻")
    gaps = times.diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError("ETTh1 本周使用的记录必须每小时一条，不能有缺口")


def make_windows(
    series: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把一条连续序列构造成单步预测样本。

    例如 series=[a,b,c,d,e]、L=3：
    - X[0]=[a,b,c]，y[0]=d，target_index[0]=3
    - X[1]=[b,c,d]，y[1]=e，target_index[1]=4

    因而长度为 N 的序列可以构造 N-L 条样本，X 的形状为 (N-L, L)。
    """
    values = np.asarray(series, dtype=float).reshape(-1)
    if window_size <= 0:
        raise ValueError(f"window_size 必须为正整数，实际为 {window_size}")
    if len(values) <= window_size:
        raise ValueError(
            f"序列长度必须大于窗口长度，当前序列 N={len(values)}、L={window_size}"
        )

    sample_count = len(values) - window_size
    x = np.empty((sample_count, window_size), dtype=float)
    y = np.empty(sample_count, dtype=float)
    target_index = np.arange(window_size, len(values), dtype=int)

    for row, target in enumerate(target_index):
        x[row] = values[target - window_size : target]
        y[row] = values[target]

    return x, y, target_index


def make_window_examples(
    series: np.ndarray,
    times: np.ndarray,
    window_size: int,
) -> pd.DataFrame:
    """把短序列窗口写成逐行可核对的表格记录。"""
    x, y, target_index = make_windows(series, window_size)
    datetime_values = pd.to_datetime(times)
    rows: list[dict[str, object]] = []
    for row, target in enumerate(target_index):
        rows.append(
            {
                "window_size": window_size,
                "input_start": int(target - window_size),
                "input_end": int(target - 1),
                "target_index": int(target),
                "target_time": datetime_values[target],
                "input_values": repr(x[row].round(6).tolist()),
                "target_value": float(y[row]),
            }
        )
    return pd.DataFrame(rows)


def split_by_target_time(
    target_times: np.ndarray,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict[str, np.ndarray]:
    """按**目标时刻**划分窗口，而不是先切序列再分别构造窗口。

    这样首条验证/测试样本仍能使用边界前已经真实观测到的 L 小时历史，
    同时模型训练时不使用验证或测试时段的目标值。
    """
    times = pd.to_datetime(target_times)
    train_mask = times < train_end
    validation_mask = (times >= train_end) & (times < validation_end)
    test_mask = (times >= validation_end) & (times < test_end)
    masks = {
        "train": np.asarray(train_mask, dtype=bool),
        "validation": np.asarray(validation_mask, dtype=bool),
        "test": np.asarray(test_mask, dtype=bool),
    }

    covered = masks["train"].astype(int) + masks["validation"].astype(int) + masks["test"].astype(int)
    if not np.all(covered == 1):
        bad_count = int(np.sum(covered != 1))
        raise ValueError(f"有 {bad_count} 个窗口目标没有被恰好分入一个数据部分")
    return masks


def select_rows(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """根据布尔掩码取出某个数据部分；单独写出可减少 X、y、时间错位。"""
    return np.asarray(values)[mask]
