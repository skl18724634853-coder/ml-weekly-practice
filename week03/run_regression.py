"""第三周：多特征线性回归与 Ridge 正则化对照。

数据来源：sklearn.datasets.load_diabetes(as_frame=True, scaled=False)。
运行后会在 results/ 中生成两份 CSV 和两张 PNG：
  * metrics_comparison.csv：五个模型的训练/验证指标及最终测试指标
  * weights_comparison.csv：三种多特征模型在标准化输入下的权重
  * rmse_train_validation.png：训练、验证 RMSE 对比图
  * weights_comparison.png：三组权重对比图
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只导出图片，不依赖图形界面
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------ 关键设置 ------------------------
SEED = 42
RIDGE_ALPHAS = (1, 100)
RESULTS_DIR = Path(__file__).parent / "results"
# ----------------------------------------------------------


def use_utf8_output() -> None:
    """避免 Windows 终端输出中文时乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """兼容不同 scikit-learn 版本的 RMSE 计算。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def split_indices(sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一次性划分样本行，得到固定约 60%/20%/20% 的三份索引。"""
    indices = np.arange(sample_count)
    train_idx, remaining_idx = train_test_split(
        indices, test_size=0.4, random_state=SEED
    )
    validation_idx, test_idx = train_test_split(
        remaining_idx, test_size=0.5, random_state=SEED
    )
    return train_idx, validation_idx, test_idx


def evaluate(
    model_name: str, y_train: np.ndarray, train_pred: np.ndarray,
    y_validation: np.ndarray, validation_pred: np.ndarray,
) -> dict[str, float | str]:
    """以完全一致的 MAE、RMSE 口径记录训练和验证集表现。"""
    return {
        "model": model_name,
        "train_MAE": mean_absolute_error(y_train, train_pred),
        "train_RMSE": rmse(y_train, train_pred),
        "validation_MAE": mean_absolute_error(y_validation, validation_pred),
        "validation_RMSE": rmse(y_validation, validation_pred),
    }


def plot_rmse(metrics: pd.DataFrame, output_path: Path) -> None:
    """绘制所有模型训练/验证 RMSE 的分组柱状图。"""
    labels = metrics["model"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, metrics["train_RMSE"], width, label="Train RMSE", color="#4C78A8")
    ax.bar(x + width / 2, metrics["validation_RMSE"], width, label="Validation RMSE", color="#F58518")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Training and validation RMSE comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_weights(weights: pd.DataFrame, output_path: Path) -> None:
    """将三组多特征模型的标准化输入权重绘制在同一张分组柱状图中。"""
    feature_names = weights["feature"].tolist()
    model_columns = ["linear_regression", "ridge_alpha_1", "ridge_alpha_100"]
    display_names = ["LinearRegression", "Ridge (alpha=1)", "Ridge (alpha=100)"]
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    x = np.arange(len(feature_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for position, (column, label, color) in enumerate(zip(model_columns, display_names, colors)):
        ax.bar(x + (position - 1) * width, weights[column], width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, feature_names)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Coefficient (standardized input)")
    ax.set_title("Coefficient comparison for multifeature models")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    use_utf8_output()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 读取原始尺度的数据。target 只作为预测目标，绝不进入 X。
    dataset = load_diabetes(as_frame=True, scaled=False)
    X = dataset.data.copy()
    y = dataset.target.to_numpy(dtype=float)
    feature_names = X.columns.tolist()
    bmi_position = feature_names.index("bmi")

    # 2. 对“样本行”一次性固定划分，再由同一索引取得 bmi 或全部特征。
    train_idx, validation_idx, test_idx = split_indices(len(X))
    X_train = X.iloc[train_idx]
    X_validation = X.iloc[validation_idx]
    X_test = X.iloc[test_idx]
    y_train, y_validation, y_test = y[train_idx], y[validation_idx], y[test_idx]

    # 3. 只在训练集拟合每一列的均值、标准差；验证/测试集只转换。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练五个模型。基线不使用任何输入特征。
    baseline_value = float(y_train.mean())
    models: dict[str, LinearRegression | Ridge] = {
        "single_feature_linear_regression": LinearRegression(),
        "multifeature_linear_regression": LinearRegression(),
        "ridge_alpha_1": Ridge(alpha=1),
        "ridge_alpha_100": Ridge(alpha=100),
    }
    models["single_feature_linear_regression"].fit(
        X_train_scaled[:, [bmi_position]], y_train
    )
    for name in ("multifeature_linear_regression", "ridge_alpha_1", "ridge_alpha_100"):
        models[name].fit(X_train_scaled, y_train)

    # 5. 统一计算训练、验证 MAE/RMSE；此时测试集尚未参与选择。
    metric_rows = [
        evaluate(
            "mean_baseline", y_train, np.full(len(y_train), baseline_value),
            y_validation, np.full(len(y_validation), baseline_value),
        )
    ]
    metric_rows.append(
        evaluate(
            "single_feature_linear_regression", y_train,
            models["single_feature_linear_regression"].predict(X_train_scaled[:, [bmi_position]]),
            y_validation,
            models["single_feature_linear_regression"].predict(X_validation_scaled[:, [bmi_position]]),
        )
    )
    for name in ("multifeature_linear_regression", "ridge_alpha_1", "ridge_alpha_100"):
        metric_rows.append(
            evaluate(
                name, y_train, models[name].predict(X_train_scaled),
                y_validation, models[name].predict(X_validation_scaled),
            )
        )
    metrics = pd.DataFrame(metric_rows)

    # 6. 严格按验证集 RMSE 在两种 Ridge 中选 alpha；相同则选较小 alpha=1。
    ridge_validation = metrics.set_index("model").loc[["ridge_alpha_1", "ridge_alpha_100"]]
    selected_ridge_name = min(
        ridge_validation.index,
        key=lambda name: (float(ridge_validation.loc[name, "validation_RMSE"]), name),
    )
    selected_alpha = 1 if selected_ridge_name == "ridge_alpha_1" else 100

    # 7. 测试集只评价基线、单特征、多特征和已选定的 Ridge。
    test_predictions: dict[str, np.ndarray] = {
        "mean_baseline": np.full(len(y_test), baseline_value),
        "single_feature_linear_regression": models["single_feature_linear_regression"].predict(
            X_test_scaled[:, [bmi_position]]
        ),
        "multifeature_linear_regression": models["multifeature_linear_regression"].predict(X_test_scaled),
        selected_ridge_name: models[selected_ridge_name].predict(X_test_scaled),
    }
    metrics["test_MAE"] = np.nan
    metrics["test_RMSE"] = np.nan
    for name, prediction in test_predictions.items():
        metrics.loc[metrics["model"] == name, "test_MAE"] = mean_absolute_error(y_test, prediction)
        metrics.loc[metrics["model"] == name, "test_RMSE"] = rmse(y_test, prediction)

    # 8. 保存指标、权重及两张图。
    metrics["selected_ridge"] = ""
    metrics.loc[metrics["model"] == selected_ridge_name, "selected_ridge"] = "yes"
    metrics = metrics.round(3)
    metrics.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig")

    weights = pd.DataFrame({
        "feature": feature_names,
        "linear_regression": models["multifeature_linear_regression"].coef_,
        "ridge_alpha_1": models["ridge_alpha_1"].coef_,
        "ridge_alpha_100": models["ridge_alpha_100"].coef_,
    }).round(6)
    weights.to_csv(RESULTS_DIR / "weights_comparison.csv", index=False, encoding="utf-8-sig")
    plot_rmse(metrics, RESULTS_DIR / "rmse_train_validation.png")
    plot_weights(weights, RESULTS_DIR / "weights_comparison.png")

    print("完成：多特征线性回归与 Ridge 对照实验")
    print(f"数据划分：训练 {len(y_train)} / 验证 {len(y_validation)} / 测试 {len(y_test)}，随机种子 {SEED}")
    print("Ridge 选择依据：验证集 RMSE")
    print(f"选定：{selected_ridge_name}（alpha={selected_alpha}）")
    print(metrics.to_string(index=False))
    print(f"结果目录：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
