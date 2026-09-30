# 第三周：多特征回归与 Ridge 正则化

本目录使用 scikit-learn 内置的 diabetes 数据集，完成均值基线、单特征线性回归、多特征线性回归，以及两种 Ridge 正则化强度的公平对照。

## 数据与关键设置

- 数据：`sklearn.datasets.load_diabetes(as_frame=True, scaled=False)`，共 442 条样本、10 个输入特征；`target` 仅作预测目标，不放入输入。
- 划分：样本行一次性随机划分为训练 60%、验证 20%、测试 20%，随机种子为 `42`。
- 标准化：`StandardScaler` 仅在训练特征上 `fit`，验证和测试特征复用相同统计量；目标值不标准化。
- 模型：均值基线、`bmi` 单特征 `LinearRegression`、全部 10 特征 `LinearRegression`、`Ridge(alpha=1)`、`Ridge(alpha=100)`。
- 选择规则：只依据验证集 RMSE 在两个 Ridge 中选择 alpha；若并列，选择较小的 `alpha=1`。测试集不用于选参，也不会在训练集与验证集合并后重训。

## 环境与运行

- 实测环境：Python 3.13（本机在 conda 环境 `ml-weekly` 中运行），
  依赖版本见 `week03/requirements.txt`（numpy 2.5.3、pandas 3.0.6、
  scikit-learn 1.9.1、matplotlib 3.11.2、scipy 1.18.1）。
- 安装依赖（在仓库根目录执行）：

```powershell
pip install -r week03\requirements.txt
```

- 运行脚本（默认写出到 `week03/results/`，缺失时自动创建）：

```powershell
python week03\run_regression.py
```

- 也可以另建独立环境后按同样两条命令安装、运行：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r week03\requirements.txt
.\.venv\Scripts\python.exe week03\run_regression.py
```

若 PowerShell 阻止激活脚本，上面的写法不需要激活，直接调用 `.venv\Scripts\python.exe` 即可。

## 文件说明

- `run_regression.py`：固定划分、训练集标准化、五个模型训练、验证集选 Ridge、最终测试评价及文件导出。
- `results/metrics_comparison.csv`：五个模型的训练/验证 MAE、RMSE；均值基线、单特征、多特征和选定 Ridge 的测试指标。`selected_ridge=yes` 标注最终选择。
- `results/weights_comparison.csv`：普通多特征线性回归、Ridge alpha=1、Ridge alpha=100 的 10 个权重。它们都对应标准化后的输入。
- `results/rmse_train_validation.png`：训练和验证 RMSE 分组柱状图。
- `results/weights_comparison.png`：三组多特征模型的权重对比图。

## 结果解读提醒

增加特征或使用 Ridge 不保证验证/测试误差必然降低。Ridge 的 `alpha` 是 L2 正则化强度，不是梯度下降的学习率；alpha 增大一般会更强地约束权重规模。权重正负表示预测方向，但不能据此直接推断因果关系或绝对特征重要性。
