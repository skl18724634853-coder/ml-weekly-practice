"""实践一入口：用 ETTh1 前 12 小时油温核对序列窗口。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from data import check_hourly_time_index, load_etth1, make_window_examples
from reporting import save_csv


RESULTS_DIR = Path(__file__).parent / "results"
SHORT_LENGTH = 12
WINDOW_OPTIONS = (3, 5)


def use_utf8_output() -> None:
    """避免 Windows 终端中的中文输出乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def print_examples(examples: pd.DataFrame) -> None:
    """打印每种窗口的首尾样本，方便回到原始表格手工核对。"""
    for length in WINDOW_OPTIONS:
        part = examples[examples["window_size"] == length]
        print(f"\nL={length}：样本数 {len(part)}（应为 N-L = {SHORT_LENGTH - length}）")
        for label, row in (("首条", part.iloc[0]), ("末条", part.iloc[-1])):
            print(
                f"  {label}：输入下标 {row['input_start']}..{row['input_end']}，"
                f"目标下标 {row['target_index']}，目标时间 {row['target_time']}，"
                f"输入 {row['input_values']} -> 目标 {row['target_value']:.3f}"
            )


def main() -> None:
    use_utf8_output()
    RESULTS_DIR.mkdir(exist_ok=True)

    frame = load_etth1()
    check_hourly_time_index(frame["date"])
    short = frame.iloc[:SHORT_LENGTH].copy()
    series = short["OT"].to_numpy(dtype=float)
    times = short["date"].to_numpy()

    print("前 12 小时的 ETTh1 油温：")
    print(short[["date", "OT"]].to_string(index=True))
    print("\n观察：油温并非单调变化；窗口会保留这种连续的升降历史。")

    all_examples = pd.concat(
        [make_window_examples(series, times, window_size=length) for length in WINDOW_OPTIONS],
        ignore_index=True,
    )
    print_examples(all_examples)
    output = RESULTS_DIR / "window_examples.csv"
    save_csv(all_examples, output)
    print(f"\n已保存：{output}")


if __name__ == "__main__":
    main()
