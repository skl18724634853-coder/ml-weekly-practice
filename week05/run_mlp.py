"""实践二入口（run_mlp.py）：训练小型 MLP 并解释核心实现

流程（组会就按这个顺序讲）：
1. 准备数据：60/20/20 划分、训练集拟合 StandardScaler、目标标准化、转张量
2. 检查前向形状：(32,10) → (32,16) → (32,16) → (32,1)
3. 两套配置各训练 200 轮：MLP+SGD(lr=0.01) 与 MLP+Adam(lr=0.001)
4. 每轮记录完整训练集 / 验证集的标准化 MSE，并保留验证最优轮次的参数
5. 恢复最佳参数，换算回原始尺度，算 MAE / RMSE
6. 与均值基线、线性回归基线对照；按验证 RMSE 选一组 MLP，只对它算测试指标
7. 输出 history.csv、metrics.csv、loss_curves.png、rmse_comparison.png
8. 隐藏层宽度 16 → 8 的前向检查

运行：进入 week05 后执行  python run_mlp.py
"""

from __future__ import annotations

import copy
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

import data
import evaluation
import models
import optimizers
import reporting
import training

# ------------------------------ 实验设置 ------------------------------
EPOCHS = 200              # 两组都训练满 200 轮
LOSS_FUNCTION = nn.MSELoss()
# ---------------------------------------------------------------------


def use_utf8_output() -> None:
    """避免 Windows 终端输出中文时乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def describe_shapes(model: nn.Module, shapes: list[tuple[int, ...]]) -> str:
    """把"输入 + 各层输出"的形状拼成一条箭头串，便于指着讲。"""
    names = ["输入"] + [type(layer).__name__ for layer in model]
    return " → ".join(f"{name}{shape}" for name, shape in zip(names, shapes))


def print_forward_check(model: nn.Module, batch: torch.Tensor, title: str) -> list[tuple[int, ...]]:
    """打印一次前向的形状变化，并返回形状列表。"""
    shapes = models.forward_shape_trace(model, batch)
    print(f"{title}")
    print(f"  {describe_shapes(model, shapes)}")
    print(f"  解释：一批有 {batch.shape[0]} 条样本，每条 {batch.shape[1]} 个特征；"
          f"隐藏层把它变成 {shapes[1][1]} 维隐藏特征；最后每条样本只有 1 个预测值。")
    print()
    return shapes


def build_metric_row(
    *,
    model_name: str,
    display_name: str,
    learning_rate: float | str,
    best_epoch: int | str,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
    selected: str = "",
    note: str = "",
) -> dict[str, object]:
    """统一一张 metrics.csv 的行；没有的值留空（NaN / 空串）。"""
    return {
        "model": model_name,
        "display_name": display_name,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "train_MAE": train_metrics["MAE"],
        "train_RMSE": train_metrics["RMSE"],
        "validation_MAE": validation_metrics["MAE"],
        "validation_RMSE": validation_metrics["RMSE"],
        "test_MAE": np.nan if test_metrics is None else test_metrics["MAE"],
        "test_RMSE": np.nan if test_metrics is None else test_metrics["RMSE"],
        "selected_mlp": selected,
        "target_scale": "original",
        "note": note,
    }


def main() -> None:
    use_utf8_output()
    reporting.ensure_results_dir()

    # ================= 1. 准备数据 =================
    print("=" * 78)
    print("实践二：训练小型 MLP（diabetes 回归）")
    print("=" * 78)
    dataset = data.prepare_diabetes()
    print(f"数据划分（种子 {data.SEED}）：训练 {len(dataset.y_train_raw)} 条 / "
          f"验证 {len(dataset.y_validation_raw)} 条 / 测试 {len(dataset.y_test_raw)} 条")
    print(f"输入特征 {dataset.n_features} 个：{dataset.feature_names}")
    print(f"目标标准化统计量（只来自训练集）：y_mean = {dataset.y_mean:.4f}，y_std = {dataset.y_std:.4f}")
    print(f"张量形状：训练特征 {tuple(dataset.x_train_tensor.shape)}，"
          f"训练标签 {tuple(dataset.y_train_scaled.shape)}")
    print("说明：目标已标准化，所以训练曲线上的 MSE 是「标准化单位」；"
          "最后汇报的 MAE / RMSE 会换算回原始单位。")
    print()

    # ================= 2. 前向形状检查（宽度 16）=================
    torch.manual_seed(data.SEED)          # 固定初始化，保证每次运行结果一致
    probe_model = models.build_mlp(hidden_dim=models.DEFAULT_HIDDEN_DIM)
    demo_batch = dataset.x_train_tensor[:32]
    print_forward_check(probe_model, demo_batch, "前向形状检查（隐藏层宽度 16，取 32 条样本）")

    # ================= 3. 两套配置一起训练 =================
    # 先创建一次初始网络，再深拷贝两份：两组从完全相同的初始参数出发，
    # 这样最后的差别只可能来自优化器（和各自的学习率）。
    torch.manual_seed(data.SEED)
    initial_model = models.build_mlp(hidden_dim=models.DEFAULT_HIDDEN_DIM)

    histories: list[dict[str, object]] = []
    best_records: dict[str, training.BestModel] = {}
    trained_models: dict[str, nn.Module] = {}
    configs = {str(config["name"]): config for config in optimizers.MLP_OPTIMIZER_CONFIGS}

    for config in optimizers.MLP_OPTIMIZER_CONFIGS:
        name = str(config["name"])
        label = str(config["label"])
        learning_rate = float(config["learning_rate"])

        model = copy.deepcopy(initial_model)          # 独立的模型副本，初始参数相同
        optimizer = optimizers.build_optimizer(
            name, model,
            learning_rate=learning_rate,
            momentum=float(config["momentum"]),
            weight_decay=float(config["weight_decay"]),
        )
        # 每组各自新建 DataLoader，用同一个打乱种子，保证每轮样本顺序一致
        loader = data.make_train_loader(dataset, batch_size=data.BATCH_SIZE, shuffle_seed=data.SEED)

        print(f"开始训练 {label}：learning_rate = {learning_rate}，"
              f"batch_size = {data.BATCH_SIZE}，epochs = {EPOCHS}")
        history, best = training.fit_mlp(
            model=model,
            loader=loader,
            data=dataset,
            epochs=EPOCHS,
            loss_fn=LOSS_FUNCTION,
            optimizer=optimizer,
            label=label,
            learning_rate=learning_rate,
        )
        histories.extend(history)
        best_records[name] = best
        trained_models[name] = model

        print(f"  完成。最佳验证 MSE = {best.validation_mse:.4f} 出现在第 {best.epoch} 轮"
              f"（最后一轮的验证 MSE = {history[-1]['validation_MSE']:.4f}）")
        print()

    history_frame = pd.DataFrame(histories)

    # ================= 4. 为什么最后一批 loss 不等于整轮训练 MSE =================
    last_epoch_rows = history_frame[history_frame["epoch"] == EPOCHS]
    print("核对：最后一批的批次损失 ≠ 这一轮结束时的训练 MSE（组会 B4）")
    for row in last_epoch_rows.itertuples(index=False):
        print(f"  {row.config:<14} 最后一批 loss = {row.last_batch_loss:.4f}，"
              f"整份训练集重新评价的 MSE = {row.train_MSE:.4f}，"
              f"相差 {abs(row.last_batch_loss - row.train_MSE):.4f}")
    print("  原因：① 最后一批只有 9 条样本（265 条不能被 32 整除），统计噪声大；")
    print("        ② 它是用「这一批更新之前」的参数算的，而整份 MSE 用的是整轮更新之后的参数。")
    print()

    # ================= 5. 恢复最佳参数，换算回原始尺度 =================
    mlp_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, best in best_records.items():
        model = trained_models[name]
        training.load_best_state(model, best)          # 换成最佳轮次的参数
        mlp_metrics[name] = {
            "train": evaluation.evaluate_mlp_raw(
                model, dataset.x_train_tensor, dataset.y_train_raw, dataset.y_mean, dataset.y_std
            ),
            "validation": evaluation.evaluate_mlp_raw(
                model, dataset.x_validation_tensor, dataset.y_validation_raw,
                dataset.y_mean, dataset.y_std,
            ),
        }

    # ================= 6. 两个基线 =================
    baseline_value = training.fit_mean_baseline(dataset.y_train_raw)
    linear_model = training.fit_linear_baseline(dataset.x_train_scaled, dataset.y_train_raw)

    baseline_metrics = {
        "train": evaluation.regression_metrics(
            dataset.y_train_raw, np.full(len(dataset.y_train_raw), baseline_value)
        ),
        "validation": evaluation.regression_metrics(
            dataset.y_validation_raw, np.full(len(dataset.y_validation_raw), baseline_value)
        ),
    }
    linear_metrics = {
        "train": evaluation.regression_metrics(
            dataset.y_train_raw, linear_model.predict(dataset.x_train_scaled)
        ),
        "validation": evaluation.regression_metrics(
            dataset.y_validation_raw, linear_model.predict(dataset.x_validation_scaled)
        ),
    }

    # ================= 7. 按验证 RMSE 选一组 MLP（相同则选 SGD）=================
    # 只依据未舍入的验证 RMSE 比较；用 (RMSE, 是否SGD) 排序实现"相同固定选 SGD"
    selected_name = min(
        ("sgd", "adam"),
        key=lambda candidate: (mlp_metrics[candidate]["validation"]["RMSE"], candidate != "sgd"),
    )

    # ================= 8. 测试集只评价三个模型 =================
    test_metrics: dict[str, dict[str, float]] = {
        "mean_baseline": evaluation.regression_metrics(
            dataset.y_test_raw, np.full(len(dataset.y_test_raw), baseline_value)
        ),
        "linear_regression": evaluation.regression_metrics(
            dataset.y_test_raw, linear_model.predict(dataset.x_test_scaled)
        ),
        f"{selected_name}_mlp": evaluation.evaluate_mlp_raw(
            trained_models[selected_name], dataset.x_test_tensor, dataset.y_test_raw,
            dataset.y_mean, dataset.y_std,
        ),
    }

    # ================= 9. 组装 metrics.csv 的四行 =================
    metric_rows = [
        build_metric_row(
            model_name="mean_baseline",
            display_name="Mean baseline",
            learning_rate="",
            best_epoch="",
            train_metrics=baseline_metrics["train"],
            validation_metrics=baseline_metrics["validation"],
            test_metrics=test_metrics["mean_baseline"],
            note=f"固定预测 = 训练目标均值 {baseline_value:.3f}",
        ),
        build_metric_row(
            model_name="linear_regression",
            display_name="Linear regression",
            learning_rate="",
            best_epoch="",
            train_metrics=linear_metrics["train"],
            validation_metrics=linear_metrics["validation"],
            test_metrics=test_metrics["linear_regression"],
            note="10 个标准化特征，原始目标",
        ),
    ]
    for config in optimizers.MLP_OPTIMIZER_CONFIGS:
        name = str(config["name"])
        label = str(config["label"])
        is_selected = name == selected_name
        metric_rows.append(build_metric_row(
            model_name=f"{name}_mlp",
            display_name=label,
            learning_rate=float(config["learning_rate"]),
            best_epoch=int(best_records[name].epoch),
            train_metrics=mlp_metrics[name]["train"],
            validation_metrics=mlp_metrics[name]["validation"],
            test_metrics=test_metrics.get(f"{name}_mlp"),   # 未被选中的那组测试列留空
            selected="yes" if is_selected else "",
            note=f"最佳轮次参数；隐藏层宽度 {models.DEFAULT_HIDDEN_DIM}",
        ))
    metrics_frame = pd.DataFrame(metric_rows)

    # ================= 10. 隐藏层宽度修改检查：16 → 8 =================
    print("隐藏层宽度修改检查（16 → 8，只做前向，不重新训练）")
    narrow_model = models.build_mlp(hidden_dim=8)
    narrow_shapes = models.forward_shape_trace(narrow_model, demo_batch)
    print(f"  宽度 8 ：{describe_shapes(narrow_model, narrow_shapes)}")
    wide_shapes = models.forward_shape_trace(initial_model, demo_batch)
    print(f"  宽度 16：{describe_shapes(initial_model, wide_shapes)}")
    print(f"  宽度 8 的参数量：{models.parameter_count(narrow_model)}")
    print(f"  宽度 16 的参数量：{models.parameter_count(initial_model)}")
    print("  要点：hidden_dim 同时控制第一层的「输出维数」和第二层的「输入维数」，")
    print("        只改第一层不衔接；宽度变了也不影响「每条样本只有一个预测值」。")
    print()

    # ================= 11. 保存 CSV 与 PNG =================
    history_path = reporting.save_history(histories)
    metrics_path = reporting.save_metrics(metric_rows)

    best_epochs = {str(config["label"]): int(best_records[str(config["name"])].epoch)
                   for config in optimizers.MLP_OPTIMIZER_CONFIGS}
    curves_path = reporting.plot_loss_curves(history_frame, best_epochs)

    # 画 RMSE 对照图时用简短的显示名
    plot_frame = metrics_frame[["display_name", "train_RMSE", "validation_RMSE"]].copy()
    rmse_path = reporting.plot_rmse_comparison(plot_frame)

    # ================= 12. 控制台汇总 =================
    print("=" * 78)
    print("结果汇总")
    print("=" * 78)
    summary = metrics_frame[[
        "display_name", "learning_rate", "best_epoch",
        "train_RMSE", "validation_RMSE", "test_RMSE", "selected_mlp",
    ]]
    reporting.print_table("指标（RMSE / MAE 均为原始目标单位）：", summary)

    print(f"按验证 RMSE 选定：{configs[selected_name]['label']}"
          f"（验证 RMSE = {mlp_metrics[selected_name]['validation']['RMSE']:.4f}）")
    print("说明：两组学习率不同，比较的是「两套具体配置」；"
          "并不把全部差异都归因于优化器算法本身。")
    print("说明：任务不要求 MLP 超过线性回归，也不要求 Adam 超过 SGD，结果如实呈现即可。")
    print()
    print("已生成文件：")
    for path in (history_path, metrics_path, curves_path, rmse_path):
        print(f"  {path}")
    print()
    print("注意：metrics.csv 中未选中那组 MLP 的测试列留空（本周只评价选定的那一组）。")


if __name__ == "__main__":
    main()
