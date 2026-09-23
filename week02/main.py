"""第二周任务：单特征线性回归 —— NumPy 梯度下降与模型评价
================================================================

任务要求（对照检查）：
  1. 划分数据：60% 训练 / 20% 验证 / 20% 测试，随机种子 42         → split_data()
  2. 标准化特征：只用训练集的均值和标准差，同一组统计量变换三个集合    → standardize()
  3. 均值基线：对每条输入都预测训练集目标均值                        → build_baseline()
  4. NumPy 梯度下降：w=0、b=0 起步，全批量，保存每轮训练损失          → gd_fit()
  5. 比较学习率：0.01 与 0.1 各 500 轮，损失曲线画在同一张图          → plot_loss_curves()
  6. 与库函数对照：同一份标准化数据上拟合 LinearRegression            → compare_with_sklearn()
  7. 评价泛化：训练/验证集 MAE、RMSE；选定 GD 配置后算测试指标并存表   → build_metrics_table()
  8. 检查预测：测试集真实值 vs 预测值散点图 + y=x 参考线，核对样本    → plot_test_scatter()

数据来源：sklearn.datasets.load_diabetes(as_frame=True, scaled=False)
         本周必做只用 bmi 一个特征，拟合 预测值 = w × bmi + b。
运行方式见 README.md。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接把图存成文件（便于无界面环境）
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================== 可调参数（旋钮） ====================
FEATURE = "bmi"       # 本周必做：只用这一个特征
SEED = 42             # 固定随机种子，保证每次划分一致
EPOCHS = 500          # 每个学习率训练的轮数
LRS = [0.01, 0.1]     # 两个对比学习率
# =======================================================

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

LINE = "=" * 62


def use_utf8_output() -> None:
    """Windows 终端默认 GBK，显式切到 UTF-8 让中文输出不乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def banner(step: str) -> None:
    print()
    print(LINE)
    print(f"【{step}】")
    print(LINE)


def load_data():
    """加载 diabetes 数据，只取 bmi 特征和 target 目标。"""
    df = load_diabetes(as_frame=True, scaled=False).frame
    X = df[[FEATURE]].to_numpy(dtype=float)   # (442, 1) 二维数组
    y = df["target"].to_numpy(dtype=float)    # (442,)   一维数组
    return X, y, df


def split_data(X, y):
    """划分 60/20/20：先留出 40%，再把 40% 均分为验证和测试。

    注意：这不是时序数据，随机划分 OK；时序任务不能这样打乱。
    """
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=SEED)      # 60% 训练，40% 暂留
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=SEED)  # 40% 均分 → 各 20%
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def standardize(X_tr, X_va, X_te):
    """只用训练集的均值/标准差做标准化，三份数据用同一组统计量。

    为什么不能对三份分别拟合？那样验证/测试集的数值尺度会和训练时
    对不上，等于在训练中"偷看"了它们的信息，评价会失真。
    """
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    return (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd, mu, sd


def build_baseline(y_tr):
    """均值基线：完全不看特征，对任何输入都预测训练集目标均值。"""
    mean_target = float(np.mean(y_tr))
    return mean_target, lambda X: np.full(X.shape[0], mean_target)


def gd_fit(X, y, lr, epochs=EPOCHS, w0=0.0, b0=0.0):
    """全批量梯度下降，拟合 y ≈ w*x + b。

    X: (n,1) 标准化后的特征；y: (n,) 目标。
    返回 (w, b, losses)，losses 是每轮的训练 MSE。
    """
    x = X[:, 0]                    # 取成一维数组，避免形状陷阱
    w, b = w0, b0
    losses = np.empty(epochs)

    for t in range(epochs):
        # ---------- ① 预测：用当前参数算每个样本的预测值 ----------
        pred = w * x + b

        # ---------- ② 计算梯度：MSE 对 w、b 的偏导 ----------
        # error = 预测 - 真实
        # MSE   = mean(error^2)
        # dw    = 2 * mean(error * x)   （链式法则：外层 2*error 内层 *x）
        # db    = 2 * mean(error)       （内层对 b 求导为 1）
        error = pred - y
        dw = 2.0 * np.mean(error * x)
        db = 2.0 * np.mean(error)

        # ---------- ③ 更新参数：同时更新，且都用更新前的梯度 ----------
        w = w - lr * dw
        b = b - lr * db

        losses[t] = float(np.mean(error ** 2))   # 保存本轮训练损失

    return w, b, losses


def rmse(y_true, y_pred):
    """RMSE = sqrt(mean((预测-真实)^2))，与目标同单位，越大越差。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def plot_loss_curves(losses_dict, path):
    """两个学习率的训练损失曲线画在同一张图。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr, losses in losses_dict.items():
        ax.plot(losses, label=f"lr = {lr}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (MSE)")
    ax.set_title(f"Gradient descent loss curves ({FEATURE} feature)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test_scatter(y_te, preds, path, mae_val, rmse_val):
    """测试集真实值 vs 预测值散点图 + y=x 参考线。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, preds, s=20, alpha=0.6, color="#1D9E75", edgecolor="none")
    lo, hi = min(y_te.min(), preds.min()), max(y_te.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#D85A30", linewidth=1.5, label="y = x")
    ax.set_xlabel("True target (test)")
    ax.set_ylabel("Predicted target (test)")
    ax.set_title(f"Test predictions: {FEATURE} linear regression\n"
                 f"MAE = {mae_val:.1f}, RMSE = {rmse_val:.1f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_with_sklearn(X_tr, y_tr):
    """在同一份标准化训练数据上拟合 LinearRegression 作对照。

    LinearRegression 使用普通最小二乘求解器，不是这里的梯度下降；
    梯度下降充分收敛时两者结果应接近。
    """
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model


def main() -> None:
    use_utf8_output()

    # ---------- 第 1 步：读数据 + 划分 ----------
    banner("第 1 步 · 数据与划分")
    X, y, df = load_data()
    print(f"数据：{df.shape[0]} 样本 × {df.shape[1]} 列，特征只用 {FEATURE}，目标 target")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data(X, y)
    print(f"划分：训练 {len(y_tr)} / 验证 {len(y_va)} / 测试 {len(y_te)}"
          f"（60/20/20，随机种子 {SEED}）")

    # ---------- 第 2 步：标准化（只用训练集统计量） ----------
    banner("第 2 步 · 训练集标准化")
    Xs_tr, Xs_va, Xs_te, mu, sd = standardize(X_tr, X_va, X_te)
    print(f"训练集 {FEATURE}：均值 {mu[0]:.3f}，标准差 {sd[0]:.3f}")
    print("验证/测试集用同一组均值和标准差变换；目标值保持原样。")

    # ---------- 第 3 步：均值基线 ----------
    banner("第 3 步 · 均值基线")
    mean_target, baseline_predict = build_baseline(y_tr)
    print(f"基线：对任何输入都预测训练集目标均值 {mean_target:.2f}（完全不看特征）")
    print("为什么还要比较它：如果连不看特征的模型都比不过，说明学到的特征关系没有用。")

    # ---------- 第 4、5 步：梯度下降 + 学习率对比 ----------
    banner("第 4、5 步 · NumPy 梯度下降与学习率对比")
    gd_results = {}
    for lr in LRS:
        w, b, losses = gd_fit(Xs_tr, y_tr, lr=lr, epochs=EPOCHS)
        final_mse = float(np.mean((w * Xs_tr[:, 0] + b - y_tr) ** 2))
        gd_results[lr] = {"w": w, "b": b, "losses": losses, "train_mse": final_mse}
        print(f"lr={lr}: 初始 w=0, b=0 → 训练 {EPOCHS} 轮后"
              f" w={w:.4f}, b={b:.4f}, 训练 MSE={final_mse:.2f}")

    plot_loss_curves({lr: r["losses"] for lr, r in gd_results.items()},
                     OUT_DIR / "loss_curves_lr_compare.png")
    print(f"损失曲线已保存：results/loss_curves_lr_compare.png（两条曲线同图）")

    # ---------- 第 6 步：库函数对照 ----------
    banner("第 6 步 · LinearRegression 对照")
    lr_model = compare_with_sklearn(Xs_tr, y_tr)
    sk_w, sk_b = float(lr_model.coef_[0]), float(lr_model.intercept_)
    sk_train_mse = float(mean_squared_error(y_tr, lr_model.predict(Xs_tr)))
    print(f"LinearRegression（最小二乘求解器）：w={sk_w:.4f}, b={sk_b:.4f},"
          f" 训练 MSE={sk_train_mse:.2f}")
    comparisons = []
    sk_pred = lr_model.predict(Xs_tr)
    for lr, r in gd_results.items():
        difference = np.abs(r['w'] * Xs_tr[:, 0] + r['b'] - sk_pred)
        comparisons.append({"model": f"gd_lr_{lr}", "w": r['w'], "b": r['b'],
                            "train_MSE": r['train_mse'],
                            "prediction_mean_abs_diff": float(difference.mean()),
                            "prediction_max_abs_diff": float(difference.max())})
        print(f"  → 梯度下降 lr={lr} 与之对比：w 差 {abs(r['w'] - sk_w):.4f},"
              f" b 差 {abs(r['b'] - sk_b):.4f}, 训练 MSE 差 {abs(r['train_mse'] - sk_train_mse):.6f},"
              f" 预测平均绝对差 {difference.mean():.8f}, 最大绝对差 {difference.max():.8f}")
    comparisons.append({"model": "sklearn_lr", "w": sk_w, "b": sk_b,
                        "train_MSE": sk_train_mse, "prediction_mean_abs_diff": 0.0,
                        "prediction_max_abs_diff": 0.0})
    pd.DataFrame(comparisons).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # ---------- 第 7 步：训练/验证指标，选定 GD 配置 ----------
    banner("第 7 步 · 训练/验证集指标")
    rows = []
    rows.append(("mean_baseline", baseline_predict(Xs_tr), baseline_predict(Xs_va), None))
    for lr in LRS:
        w, b = gd_results[lr]["w"], gd_results[lr]["b"]
        rows.append((f"gd_lr_{lr}", w * Xs_tr[:, 0] + b, w * Xs_va[:, 0] + b, None))
    rows.append(("sklearn_lr", lr_model.predict(Xs_tr), lr_model.predict(Xs_va), None))

    print(f"{'模型':<16}{'训练MAE':>10}{'训练RMSE':>10}{'验证MAE':>10}{'验证RMSE':>10}")
    val_rmse = {}
    for name, p_tr, p_va, _ in rows:
        r_tr, r_va = rmse(y_tr, p_tr), rmse(y_va, p_va)
        if name.startswith("gd_"):
            val_rmse[name] = r_va
        print(f"{name:<16}{mae(y_tr, p_tr):>10.2f}{r_tr:>10.2f}{mae(y_va, p_va):>10.2f}{r_va:>10.2f}")

    chosen = min(val_rmse, key=val_rmse.get)
    print("验证 RMSE（更多小数，避免舍入掩盖差异）：", val_rmse)
    print(f"→ 依据验证集 RMSE 选定梯度下降配置：{chosen}（验证 RMSE={val_rmse[chosen]:.6f}）")
    chosen_lr = float(chosen.split("_")[-1])
    w_c, b_c = gd_results[chosen_lr]["w"], gd_results[chosen_lr]["b"]

    # ---------- 第 8 步：测试集指标（只评一次，不再反复调） ----------
    banner("第 8 步 · 测试集指标与结果表")
    test_rows = [
        ("mean_baseline", baseline_predict(Xs_te), np.full(len(y_te), mean_target)),
        (f"gd_lr_{chosen_lr}", w_c * Xs_te[:, 0] + b_c, w_c * Xs_te[:, 0] + b_c),
        ("sklearn_lr", lr_model.predict(Xs_te), lr_model.predict(Xs_te)),
    ]
    print(f"{'模型':<16}{'测试MAE':>10}{'测试RMSE':>10}")
    test_metrics = {}
    for name, p_te, _ in test_rows:
        r_te = rmse(y_te, p_te)
        test_metrics[name] = (mae(y_te, p_te), r_te)
        print(f"{name:<16}{mae(y_te, p_te):>10.2f}{r_te:>10.2f}")

    # 汇总成 CSV 表格（训练+验证列全有；测试列只填被选中的三个模型）
    table = pd.DataFrame(
        [{"model": n, "train_MAE": mae(y_tr, p_tr), "train_RMSE": rmse(y_tr, p_tr),
          "val_MAE": mae(y_va, p_va), "val_RMSE": rmse(y_va, p_va)}
         for n, p_tr, p_va, _ in rows])
    test_tbl = pd.DataFrame(
        [{"model": n, "test_MAE": m, "test_RMSE": r} for n, (m, r) in test_metrics.items()])
    table = table.merge(test_tbl, on="model", how="left").round(2)
    csv_path = OUT_DIR / "metrics_comparison.csv"
    table.to_csv(csv_path, index=False)
    print(f"比较表已保存：results/metrics_comparison.csv")
    print(table.to_string(index=False))

    # ---------- 第 9 步：测试集散点图 + 样本核对 ----------
    banner("第 9 步 · 测试集预测检查")
    gd_te_pred = w_c * Xs_te[:, 0] + b_c
    pd.DataFrame({"true_target": y_te, "prediction": gd_te_pred,
                  "error": gd_te_pred - y_te,
                  "absolute_error": np.abs(gd_te_pred - y_te)}).to_csv(
                      OUT_DIR / "test_predictions.csv", index=False)
    mae_te, rmse_te = test_metrics[f"gd_lr_{chosen_lr}"]
    plot_test_scatter(y_te, gd_te_pred, OUT_DIR / "test_predictions_scatter.png",
                      mae_te, rmse_te)
    print(f"散点图已保存：results/test_predictions_scatter.png（含 y=x 参考线）")
    print(f"选定模型 {chosen}：w={w_c:.4f}, b={b_c:.4f}")
    print("抽查 5 条测试样本（真实值 / 预测值 / 误差）：")
    for i in np.random.default_rng(7).choice(len(y_te), size=5, replace=False):
        print(f"  真实 {y_te[i]:8.1f}   预测 {gd_te_pred[i]:8.1f}   误差 {gd_te_pred[i]-y_te[i]:+7.1f}")
    print("提示：训练损失下降只说明在训练数据上拟合得更好；对新数据预测是否准，")
    print("      要由验证/测试指标来判断——这是'拟合'和'泛化'的区别。")

    print(LINE)
    print("全部完成。图表与表格在 results/ 文件夹。")
    print(LINE)


if __name__ == "__main__":
    main()
