"""实践二：diabetes 数据集探索
============================================================

任务要求：
  1. 查看行列数、列名、前几行；指出哪些列是特征、哪列是目标  → 第 1 步
  2. 统计各列缺失值 + 数值列基本统计                          → 第 2 步
  3. 画目标值分布图 + 一个特征与目标的散点图（标明横纵轴、保存）→ 第 3 步
  4. 做一次有明确目的的小修改，先写预期再运行检查              → 见本文件顶部两个旋钮
  5. 解释从加载到出图的步骤；记录一个观察和一个不能下的结论    → 见 README / 笔记

本脚本只做数据探索，**不训练模型**（本周要求）。
运行方式见 README.md。跑一次会做两件事：
  · 在屏幕上打印结构化的文字结果（可直接用于组会展示）
  · 把两张图存进 outputs/ 文件夹

运行 --feature s3 生成第二种特征的图，和默认 bmi 图对照。
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def maybe_relaunch_with_project_venv() -> None:
    """自动切到项目 .venv；如果当前 Python 缺依赖，就直接用项目环境重跑。

    这样从 VS Code / PowerShell 直接运行脚本时，也能避免因为全局解释器缺库而报错。
    """
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if os.environ.get("DIABETES_VENV_RELAUNCHED") == "1":
        return

    if not venv_python.exists():
        return

    try:
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
        from sklearn.datasets import load_diabetes  # noqa: F401
        return
    except ModuleNotFoundError:
        print("当前 Python 环境缺少项目依赖，正在自动切换到项目虚拟环境...", file=sys.stderr)
        env = os.environ.copy()
        env["DIABETES_VENV_RELAUNCHED"] = "1"
        raise SystemExit(subprocess.call([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


maybe_relaunch_with_project_venv()

import matplotlib
matplotlib.use("Agg")            # 不弹窗，直接把图存成文件（更稳，也便于无界面环境）
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes

# ============================================================
# 第 4 步「修改实验」的两个旋钮 —— 这是全脚本唯一需要改的地方
#
#   改之前，先在笔记里写下你的预期：改了之后图会变成什么样？
#   改完按 Ctrl+S 保存，再用同一条命令重跑，然后对比「预期 vs 实际」。
#
#   ⚠️ 图名里带了特征名（02_scatter_<特征>.png），所以换特征不会覆盖旧图，
#      两张图可以并排对比 —— 这是能做这个实验的前提。
#
#   组会展示小技巧：不用改代码也能演示对比，直接跑
#       .\.venv\Scripts\python.exe diabetes_explore.py --feature s3
#   （也可以命令行临时指定：python diabetes_explore.py --feature s3）
# ============================================================
SCATTER_FEATURE = "bmi"          # 散点图横轴用哪个特征，可选：bmi / s3 / s5 / bp / age ...
HIST_BINS = 30                   # 直方图的分箱数，可以改 10 或 50 试试
# ============================================================

# 输出目录（图片存这里）
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

LINE = "=" * 62


def use_utf8_output() -> None:
    """让中文能正常显示。

    背景：Windows 终端默认编码是 GBK，而本文件的注释和输出都是 UTF-8，
    两边对不上时中文会变成 `�� 1 ����` 这样的乱码 —— 看着像程序坏了，其实只是显示问题。
    这里显式把输出编码切到 UTF-8，保证屏幕上读得出来
"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass  # 个别环境不支持就跳过，不影响功能


def banner(step: str) -> None:
    """打印一个醒目的分节标题，方便对着屏幕一步步讲。"""
    print()
    print(LINE)
    print(f"【{step}】")
    print(LINE)


def load_data() -> pd.DataFrame:
    """第 1 步的准备：把数据读成一张 pandas 表格。

    sklearn 自带 diabetes 数据集，不需要自己去网上下载。
      as_frame=True  -> 直接返回 pandas 表格（DataFrame），而不是一堆裸数组
      scaled=False   -> 用原始数值，而不是"每个数都除以标准差"以后的版本
                        （标准化之后 bmi 就不再是真实的 bmi 值，图不好解释）
    """
    data = load_diabetes(as_frame=True, scaled=False)
    return data.frame


def step1_look_at_data(df: pd.DataFrame) -> list:
    """第 1 步：查看行列数、列名、前几行，指出哪些是特征、哪列是目标。"""
    banner("第 1 步 · 数据长什么样")

    print(f"行列数：{df.shape[0]} 个样本（行） × {df.shape[1]} 列")
    print(f"列名  ：{list(df.columns)}")
    print()
    print("前 5 行（`.head()`，看数据具体长什么样）：")
    print(df.head().to_string())

    feature_cols = [c for c in df.columns if c != "target"]
    print()
    print("哪些是特征、哪列是目标：")
    print(f"  特征（输入，共 {len(feature_cols)} 个）：{feature_cols}")
    print("  目标（答案，共 1 个）   ：target")
    print("  —— target 是「一年后病情进展的定量指标」，数值越大表示该病情进展指标越高。")
    return feature_cols


def step2_missing_and_stats(df: pd.DataFrame) -> None:
    """第 2 步：统计各列缺失值 + 数值列基本统计（没有缺失也要如实记录）。"""
    banner("第 2 步 · 缺失值与基本统计")

    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    print(f"缺失值总数：{total_missing}")
    if total_missing == 0:
        print("→ 各列缺失值都是 0：这份数据是完整的，没有任何空格子。")
        print("  （学长要求「没有缺失也如实记录」——所以这里要明确写出来，不是没什么可写。）")
    else:
        print("各列缺失值：")
        print(missing[missing > 0].to_string())

    print()
    print("数值列基本统计（count 个数 / mean 均值 / std 标准差 / min 最小 /")
    print("                25% 下四分位 / 50% 中位数 / 75% 上四分位 / max 最大）：")
    print(df.describe().T.round(2).to_string())

    t = df["target"]
    print()
    print("单独看目标列 target：")
    print(f"  平均值 {t.mean():.1f}，中位数 {t.median():.1f}，最大 {t.max():.0f}")
    if t.mean() > t.median():
        print("  → 平均值比中位数大，提示可能存在右偏，应结合直方图观察，不能仅凭均值和中位数判定。")


def step3_plots(df: pd.DataFrame, feature: str) -> tuple:
    """第 3 步：画目标值分布图 + 一个特征与目标的散点图，标明横纵轴并保存。

    feature 参数就是要画的那个特征名。正常跑用默认的 bmi；
    想演示「修改实验」，就把它换成 s3 ，可通过 --feature s3 指定。
    """
    banner("第 3 步 · 画图")

    # ---------- 图一：目标值分布直方图 ----------
    # hist 内部做三件事：① 把 442 个 target 值切成 HIST_BINS 个区间
    #                    ② 数每个区间里落了多少个病人
    #                    ③ 用柱子的高度表示这个数量
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["target"], bins=HIST_BINS, color="#4C78A8", edgecolor="white", linewidth=0.6)
    ax.axvline(df["target"].mean(), color="#D85A30", linestyle="--", linewidth=1.5,
               label=f"mean = {df['target'].mean():.1f}")
    ax.set_xlabel("Target: disease progression (higher = worse)")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Distribution of the target value (n={len(df)}, bins={HIST_BINS})")
    ax.legend()
    fig.tight_layout()
    f1 = OUT_DIR / "01_target_distribution.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)                                    # 关掉画布，避免图越堆越多
    print(f"图一已保存：{f1.name}")
    print(f"  横轴 = 病情进展指标（越大越严重）    纵轴 = 病人数量（每个区间里有多少人）")
    print(f"  直方图把数值切成 {HIST_BINS} 个区间；橙色虚线是平均值 {df['target'].mean():.1f}")

    # ---------- 图二：特征 vs 目标的散点图 ----------
    # scatter 只做一件事：每个样本画一个点 —— 横坐标放特征值，纵坐标放 target
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df[feature], df["target"], s=18, alpha=0.6,
               color="#1D9E75", edgecolor="none")
    r = df[feature].corr(df["target"])                # 皮尔逊相关系数（不是画上去的，是算出来的）
    ax.set_xlabel(f"{feature} (feature)")
    ax.set_ylabel("Target: disease progression (higher = worse)")
    ax.set_title(f"{feature} vs target  (Pearson r = {r:.2f})")
    fig.tight_layout()
    f2 = OUT_DIR / "02_scatter_{}.png".format(feature)
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"图二已保存：{f2.name}")
    print(f"  横轴 = {feature}（特征）    纵轴 = 病情进展指标")
    print(f"  每个点 = 一个病人，442 个样本就是 442 个点")
    print(f"  相关系数 r = {r:.2f}"
          f"（{'正相关：两者同向变化' if r > 0 else '负相关：两者反向变化'}）")

    return f1, f2, r


def summarize(df: pd.DataFrame, feature: str, r: float) -> None:
    """收尾：把这一次运行的结果汇总成几句话，方便直接念/写进笔记。"""
    banner("本次运行的汇总")

    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values(ascending=False)
    print(f"散点图用的特征：{feature}，它与 target 的相关系数 r = {r:.2f}")
    print()
    print("全部特征与 target 的相关系数（按绝对值从强到弱，供写「观察」时参考）：")
    for name, value in corr.reindex(corr.abs().sort_values(ascending=False).index).items():
        print(f"  {name:<6} {value:+.3f}")

    print("提醒：相关系数只说明「一起变化的程度」，说明不了「谁导致谁」。")
    print("      图上 bmi 和 target 一起变大，但不能据此说 bmi 高导致了病情重 ——")
    print("      也可能有第三个因素同时影响了这两者。这就是「相关不等于因果」。")

    print(LINE)
    print(f"全部完成。图片在 {OUT_DIR.name}/ 文件夹里。")
    print(LINE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="diabetes 数据探索（只认识数据，不训练模型）")
    parser.add_argument("--feature", default=SCATTER_FEATURE,
                        help=f"散点图用哪个特征（默认 {SCATTER_FEATURE}）")
    args = parser.parse_args()

    use_utf8_output()                 # 先解决中文显示，保证输出读得出来

    df = load_data()
    step1_look_at_data(df)
    step2_missing_and_stats(df)
    _f1, _f2, r = step3_plots(df, args.feature)
    summarize(df, args.feature, r)


if __name__ == "__main__":
    main()
