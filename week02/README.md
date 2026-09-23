# 第二周：单特征线性回归与梯度下降

数据来自 scikit-learn 内置 `load_diabetes(as_frame=True, scaled=False)`，共 442 个样本；仅使用 bmi 预测一年后病情进展的定量指标。模型输入是按训练集统计量标准化后的 bmi：`预测 = w * x_standardized + b`。

## 文件与环境

- `main.py`：划分、标准化、基线、NumPy 梯度下降、库函数对照、评价和绘图。
- `requirements.txt`：精确依赖版本；Python 3.13。
- `results/`：图表、模型对照与预测明细。

## 安装与运行（Windows PowerShell）

从仓库根目录执行；若在 `week01`，先执行 `cd ..`。

```powershell
cd week02
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

使用其他 Python 3.13 安装时，可将 `py -3.13` 替换为该解释器路径。macOS/Linux 使用 `python3.13 -m venv .venv`，后续使用 `.venv/bin/python`。
VS Code 打开项目文件夹后，选择装有依赖的解释器。本机已有的 `ml-weekly` Conda 环境也可使用，但复现不依赖此环境名称。

## 实验设置

- 随机种子 42，训练/验证/测试为 265/88/89 条，约 60%/20%/20%，所有模型共用划分。
- 只用训练集计算 bmi 均值和标准差，三部分共用统计量；目标不标准化。
- 基线只使用训练集目标均值。
- 全批量梯度下降：w、b 从零开始，学习率 0.01 和 0.1，各 500 轮；同一轮梯度使用更新前参数。
- 曲线记录每次更新前的 MSE；最终对照重新计算最终参数的 MSE。
- `LinearRegression` 使用最小二乘求解器，不是这里手写的梯度下降。
- 按验证 RMSE 选择配置，再评价测试集，不根据测试分数调参。

## 结果文件

| 文件 | 内容 |
|---|---|
| `results/loss_curves_lr_compare.png` | 两种学习率的训练损失曲线 |
| `results/test_predictions_scatter.png` | 测试集真实值与预测值，含 y=x 参考线 |
| `results/metrics_comparison.csv` | 训练、验证、测试 MAE 与 RMSE |
| `results/model_comparison.csv` | 权重、截距、最终训练 MSE、相对库函数的预测差异 |
| `results/test_predictions.csv` | 测试真实值、预测值、有符号和绝对误差 |

终端抽查 5 条测试预测。未选定的梯度下降模型测试列留空，表示没有用于最终测试评价。
