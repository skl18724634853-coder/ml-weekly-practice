"""第四周：PCA 降维回归对照与 K-means 无监督聚类。

数据：scikit-learn 内置 diabetes 数据集的 10 个原始特征。
本脚本一次性完成：
1. 训练集标准化、PCA 2/5 维与线性回归对照；
2. 解释方差、标准化特征空间中的重构误差；
3. 在同一份 10 维标准化训练特征上比较 K=2 和 K=3 的 K-means。

所有结果写入本目录的 results/。运行方式见 README.md。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只保存图表，不依赖本机图形界面
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------ 实验设置 ------------------------------
SEED = 42
PCA_COMPONENT_OPTIONS = (2, 5)
K_VALUES = (2, 3)
KMEANS_N_INIT = 10
RESULTS_DIR = Path(__file__).parent / "results"
# ---------------------------------------------------------------------


def use_utf8_output() -> None:
    """避免 Windows 终端输出中文时乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差；它和目标 y 的单位相同，越小越好。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def split_indices(sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """固定随机种子，先留出 40%，再平分得到约 60%/20%/20%。"""
    indices = np.arange(sample_count)
    train_idx, remaining_idx = train_test_split(
        indices, test_size=0.4, random_state=SEED
    )
    validation_idx, test_idx = train_test_split(
        remaining_idx, test_size=0.5, random_state=SEED
    )
    return train_idx, validation_idx, test_idx


def metric_row(
    model: str,
    pca_components: int | str,
    y_train: np.ndarray,
    train_prediction: np.ndarray,
    y_validation: np.ndarray,
    validation_prediction: np.ndarray,
) -> dict[str, float | int | str]:
    """统一记录一个模型的训练集、验证集 MAE 与 RMSE。"""
    return {
        "model": model,
        "pca_components": pca_components,
        "train_MAE": float(mean_absolute_error(y_train, train_prediction)),
        "train_RMSE": rmse(y_train, train_prediction),
        "validation_MAE": float(mean_absolute_error(y_validation, validation_prediction)),
        "validation_RMSE": rmse(y_validation, validation_prediction),
    }


def reconstruction_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """在标准化后的 10 维输入空间中，对全部行、全部列求平均平方差。"""
    return float(np.mean((original - reconstructed) ** 2))


def plot_rmse_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """绘制四个回归模型训练/验证 RMSE 的分组柱状图。"""
    labels = metrics["display_name"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width / 2, metrics["train_RMSE"], width,
           color="#4C78A8", label="Train RMSE")
    ax.bar(x + width / 2, metrics["validation_RMSE"], width,
           color="#F58518", label="Validation RMSE")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("PCA regression: training and validation RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cumulative_variance(full_pca: PCA, output_path: Path) -> None:
    """绘制 1 至 10 个主成分的累计解释方差，并突出 2、5 维。"""
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    dimensions = np.arange(1, len(cumulative) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(dimensions, cumulative, marker="o", color="#4C78A8",
            label="Cumulative explained variance")
    for components, color in ((2, "#F58518"), (5, "#54A24B")):
        ratio = cumulative[components - 1]
        ax.axvline(components, color=color, linestyle="--", alpha=0.8)
        ax.scatter([components], [ratio], color=color, zorder=3,
                   label=f"{components} components: {ratio:.3f}")
    ax.set_xticks(dimensions)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of retained principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA cumulative explained variance on training features")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cluster_comparison(
    projected_train: np.ndarray,
    cluster_views: dict[int, dict[str, np.ndarray | float]],
    output_path: Path,
) -> None:
    """用同一份 PC1/PC2 投影并排展示 10 维空间中得到的 K=2、K=3 聚类。"""
    x_min, x_max = projected_train[:, 0].min(), projected_train[:, 0].max()
    y_min, y_max = projected_train[:, 1].min(), projected_train[:, 1].max()
    x_pad = max((x_max - x_min) * 0.08, 0.1)
    y_pad = max((y_max - y_min) * 0.08, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, k in zip(axes, K_VALUES):
        labels = cluster_views[k]["labels"]
        centers = cluster_views[k]["centers_projected"]
        silhouette = float(cluster_views[k]["silhouette"])
        assert isinstance(labels, np.ndarray)
        assert isinstance(centers, np.ndarray)

        for cluster_id in range(k):
            mask = labels == cluster_id
            ax.scatter(projected_train[mask, 0], projected_train[mask, 1],
                       s=24, alpha=0.68, color=cmap(cluster_id),
                       edgecolor="none", label=f"Cluster {cluster_id}")
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=170,
                   color="black", edgecolor="white", linewidth=0.9,
                   label="Projected center", zorder=4)
        ax.set_title(f"K={k}, silhouette={silhouette:.3f}\n10D clustering, 2D PCA display")
        ax.set_xlabel("PC1")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("PC2")
    fig.suptitle("K-means cluster comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    use_utf8_output()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 读取未缩放的原始 10 个输入特征；target 只用于后面的监督回归。
    dataset = load_diabetes(as_frame=True, scaled=False)
    X = dataset.data.copy()
    y = dataset.target.to_numpy(dtype=float)
    feature_names = X.columns.tolist()

    # 2. 所有模型共用完全相同的样本行划分。
    train_idx, validation_idx, test_idx = split_indices(len(X))
    X_train = X.iloc[train_idx].copy()
    X_validation = X.iloc[validation_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_validation = y[validation_idx]
    y_test = y[test_idx]

    # 3. StandardScaler 只在训练集 fit；验证/测试集只使用训练集得到的统计量 transform。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # 4. 训练均值基线与保留全部 10 个标准化特征的线性回归。
    baseline_value = float(y_train.mean())
    full_linear_model = LinearRegression().fit(X_train_scaled, y_train)

    # 5. 在标准化训练特征上拟合 PCA 2 维、5 维，并在主成分坐标上训练线性回归。
    pca_runs: dict[int, dict[str, PCA | LinearRegression | np.ndarray | float]] = {}
    pca_summary_rows: list[dict[str, float | int]] = []
    for components in PCA_COMPONENT_OPTIONS:
        pca = PCA(n_components=components, svd_solver="full", whiten=False)
        Z_train = pca.fit_transform(X_train_scaled)
        Z_validation = pca.transform(X_validation_scaled)
        Z_test = pca.transform(X_test_scaled)
        pca_linear_model = LinearRegression().fit(Z_train, y_train)

        train_reconstructed = pca.inverse_transform(Z_train)
        validation_reconstructed = pca.inverse_transform(Z_validation)
        explained_variance = float(pca.explained_variance_ratio_.sum())
        train_reconstruction = reconstruction_mse(X_train_scaled, train_reconstructed)
        validation_reconstruction = reconstruction_mse(
            X_validation_scaled, validation_reconstructed
        )

        pca_runs[components] = {
            "pca": pca,
            "Z_train": Z_train,
            "Z_validation": Z_validation,
            "Z_test": Z_test,
            "model": pca_linear_model,
            "explained_variance": explained_variance,
            "train_reconstruction_MSE": train_reconstruction,
            "validation_reconstruction_MSE": validation_reconstruction,
        }
        pca_summary_rows.append({
            "pca_components": components,
            "cumulative_explained_variance_ratio": explained_variance,
            "train_reconstruction_MSE": train_reconstruction,
            "validation_reconstruction_MSE": validation_reconstruction,
        })

    # 6. 统一计算四组模型的训练、验证指标；此时不接触测试集。
    metrics_rows = [
        metric_row(
            "mean_baseline", "", y_train,
            np.full(len(y_train), baseline_value), y_validation,
            np.full(len(y_validation), baseline_value),
        ),
        metric_row(
            "full_10_feature_linear_regression", "", y_train,
            full_linear_model.predict(X_train_scaled), y_validation,
            full_linear_model.predict(X_validation_scaled),
        ),
    ]
    for components in PCA_COMPONENT_OPTIONS:
        run = pca_runs[components]
        model = run["model"]
        Z_train = run["Z_train"]
        Z_validation = run["Z_validation"]
        assert isinstance(model, LinearRegression)
        assert isinstance(Z_train, np.ndarray)
        assert isinstance(Z_validation, np.ndarray)
        metrics_rows.append(metric_row(
            f"pca_{components}_linear_regression", components, y_train,
            model.predict(Z_train), y_validation, model.predict(Z_validation),
        ))
    metrics = pd.DataFrame(metrics_rows)

    # 7. 只根据未舍入的验证 RMSE 在两种 PCA 回归中选择；若相同，选 2 维。
    selected_components = min(
        PCA_COMPONENT_OPTIONS,
        key=lambda components: (
            float(metrics.loc[
                metrics["model"] == f"pca_{components}_linear_regression",
                "validation_RMSE",
            ].iloc[0]),
            components,
        ),
    )

    # 8. 测试集只评价基线、未降维回归和已经根据验证集选定的 PCA 回归。
    test_predictions: dict[str, np.ndarray] = {
        "mean_baseline": np.full(len(y_test), baseline_value),
        "full_10_feature_linear_regression": full_linear_model.predict(X_test_scaled),
    }
    selected_run = pca_runs[selected_components]
    selected_model = selected_run["model"]
    selected_Z_test = selected_run["Z_test"]
    assert isinstance(selected_model, LinearRegression)
    assert isinstance(selected_Z_test, np.ndarray)
    test_predictions[f"pca_{selected_components}_linear_regression"] = selected_model.predict(
        selected_Z_test
    )

    metrics["test_MAE"] = np.nan
    metrics["test_RMSE"] = np.nan
    metrics["selected_pca"] = ""
    selected_model_name = f"pca_{selected_components}_linear_regression"
    metrics.loc[metrics["model"] == selected_model_name, "selected_pca"] = "yes"
    for model_name, prediction in test_predictions.items():
        mask = metrics["model"] == model_name
        metrics.loc[mask, "test_MAE"] = mean_absolute_error(y_test, prediction)
        metrics.loc[mask, "test_RMSE"] = rmse(y_test, prediction)

    # 9. 用全部 10 个主成分展示解释方差的完整趋势；同样只在训练集上 fit。
    full_pca = PCA(n_components=len(feature_names), svd_solver="full", whiten=False)
    full_pca.fit(X_train_scaled)

    # 10. K-means 只使用训练特征 X_train_scaled，不使用回归 target。
    pca_2d = pca_runs[2]["pca"]
    Z_train_2d = pca_runs[2]["Z_train"]
    assert isinstance(pca_2d, PCA)
    assert isinstance(Z_train_2d, np.ndarray)
    cluster_metrics_rows: list[dict[str, float | int]] = []
    cluster_summary_rows: list[dict[str, float | int]] = []
    cluster_views: dict[int, dict[str, np.ndarray | float]] = {}

    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=KMEANS_N_INIT)
        labels = kmeans.fit_predict(X_train_scaled)
        silhouette = float(silhouette_score(X_train_scaled, labels, metric="euclidean"))
        cluster_metrics_rows.append({
            "K": k,
            "silhouette_score": silhouette,
            "random_state": SEED,
            "n_init": KMEANS_N_INIT,
            "fit_feature_dimensions": X_train_scaled.shape[1],
        })

        # labels 的行顺序与 X_train 保持一致；此处用原始 bmi、bp 描述每一组。
        cluster_data = X_train[["bmi", "bp"]].copy().reset_index(drop=True)
        cluster_data["cluster"] = labels
        group_summary = (
            cluster_data.groupby("cluster", sort=True)
            .agg(sample_count=("cluster", "size"),
                 mean_original_bmi=("bmi", "mean"),
                 mean_original_bp=("bp", "mean"))
            .reset_index()
        )
        assert int(group_summary["sample_count"].sum()) == len(X_train)
        for row in group_summary.itertuples(index=False):
            cluster_summary_rows.append({
                "K": k,
                "cluster": int(row.cluster),
                "sample_count": int(row.sample_count),
                "mean_original_bmi": float(row.mean_original_bmi),
                "mean_original_bp": float(row.mean_original_bp),
            })

        # 聚类中心原本位于 10 维标准化空间，用同一个训练集 PCA 投影后再绘图。
        cluster_views[k] = {
            "labels": labels,
            "centers_projected": pca_2d.transform(kmeans.cluster_centers_),
            "silhouette": silhouette,
        }

    # 11. 写出任务要求的四份 CSV 和三张 PNG。
    metrics["display_name"] = [
        "Mean baseline", "Full 10D LR", "PCA 2 + LR", "PCA 5 + LR",
    ]
    metrics = metrics.round(6)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pca_summary_rows).round(6).to_csv(
        RESULTS_DIR / "pca_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cluster_metrics_rows).round(6).to_csv(
        RESULTS_DIR / "cluster_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cluster_summary_rows).round(6).to_csv(
        RESULTS_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig"
    )
    plot_rmse_comparison(metrics, RESULTS_DIR / "rmse_comparison.png")
    plot_cumulative_variance(full_pca, RESULTS_DIR / "pca_variance.png")
    plot_cluster_comparison(Z_train_2d, cluster_views, RESULTS_DIR / "cluster_comparison.png")

    print("完成：第四周 PCA 降维回归对照与 K-means 聚类")
    print(
        f"数据划分：训练 {len(y_train)} / 验证 {len(y_validation)} / 测试 {len(y_test)}，"
        f"随机种子 {SEED}"
    )
    print(f"PCA 选择依据：验证 RMSE；选定 {selected_components} 个主成分")
    print(metrics.drop(columns="display_name").to_string(index=False))
    print(pd.DataFrame(pca_summary_rows).round(6).to_string(index=False))
    print(pd.DataFrame(cluster_metrics_rows).round(6).to_string(index=False))
    print(f"结果目录：{RESULTS_DIR}")


if __name__ == "__main__":
    main()
