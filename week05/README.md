# 第五周：MLP 与 PyTorch 训练基础

本周把前四周的数学与算法知识接到 PyTorch 上：用两个小实践把"前向计算 → 损失 → 反向传播 → 参数更新"这条链走通，并把 MLP 的训练流程按组件拆开。

两个实践共用 `load_diabetes(as_frame=True, scaled=False)` 数据：442 条样本、10 个输入特征，`target` 为回归目标。任务为回归，损失统一使用 `nn.MSELoss()`。

## 文件说明

| 文件 | 职责 |
|---|---|
| `data.py` | 实践一的两个样本；实践二的划分、标准化、张量与 DataLoader |
| `models.py` | 置零的 `nn.Linear(1,1)`；可配置隐藏层宽度的 MLP；前向形状跟踪 |
| `optimizers.py` | 按名字创建 SGD / Adam；两套 MLP 优化器配置集中在此 |
| `training.py` | 一次更新（含三阶段记录）、一批 / 一轮 / 完整训练、最佳轮次记录、两个基线拟合 |
| `evaluation.py` | 纯预测与指标计算（`eval()` + `no_grad()`，不反传、不更新参数）、尺度还原 |
| `reporting.py` | 三份 CSV 与两张 PNG 的生成 |
| `run_step.py` | 实践一入口：组织一次更新演示 |
| `run_mlp.py` | 实践二入口：组织基线与两组 MLP 实验 |
| `results/` | 运行后生成的三份 CSV 与两张 PNG |

组件之间只通过函数参数与返回值传递数据；导入组件文件不会自动开始训练。

## 环境、安装与运行

已验证环境：Python 3.13.15、torch 2.14.1+cpu、numpy 2.5.3、pandas 3.0.6、scikit-learn 1.9.1、matplotlib 3.11.2、scipy 1.18.1。CPU 即可完成，不要求 GPU。

```powershell
# 从仓库根目录（已有可用环境时，只跑最后两条即可）
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r week05\requirements.txt

# 实践一：一次梯度更新
cd week05
python run_step.py

# 实践二：训练 MLP
python run_mlp.py
```

两个入口都会自动创建 `week05/results/`。

## 给初学者的逐行讲解

不建议一口气看完全部源代码。下面四份材料从最小的两样本更新开始，逐步走到完整 MLP；每份都标出源文件位置，并解释关键单词、符号、变量、输入与输出。

1. [00_阅读说明与符号字典](学习资料/00_阅读说明与符号字典.md)：先认识 Python 中反复出现的单词和符号。
2. [01_实践一：一次更新逐行讲解](学习资料/01_实践一_一次更新逐行讲解.md)：看懂 `zero_grad()`、`backward()`、`step()` 的区别，并手算核对数字。
3. [02_数据和模型逐行讲解](学习资料/02_数据和模型逐行讲解.md)：看懂数据划分、标准化、张量形状、ReLU 与隐藏层宽度。
4. [03_训练、评价与 Adam 逐行讲解](学习资料/03_训练评价与Adam逐行讲解.md)：看懂小批量、反向传播、Adam、整轮评价与最佳参数。
5. [04_入口、输出与组会讲解](学习资料/04_入口输出与组会讲解.md)：看懂所有组件怎样连接、五份结果怎么读，以及组会可以怎样说明。

## 实践一设置

- 样本 `x=[[1.0],[2.0]]`、`y=[[2.0],[4.0]]`，均为 `float32` 张量，形状 `(2, 1)`。
- 模型 `nn.Linear(1, 1)`，权重与偏置用 `torch.no_grad()` 置零，便于手算核对。
- 优化器 `SGD(lr=0.1, momentum=0, weight_decay=0)`，损失 `nn.MSELoss()`。
- 本小例子不划分训练/验证/测试集，只用两个样本走一次更新。

跑出的数值：初始预测全为 0，初始 MSE = 10；反传得到 `weight_grad = -10`、`bias_grad = -6`；更新后 `weight = 1.0`、`bias = 0.6`，重新前向后 MSE = 1.06。脚本中的 `assert` 会核对"新参数 = 旧参数 − 0.1 × 梯度"。

## 实践二设置

- 划分：随机种子 42，约 60% / 20% / 20%，即训练 265 条、验证 88 条、测试 89 条。所有模型共用同一份划分。
- 输入标准化：`StandardScaler` 只在训练集 `fit`，验证、测试沿用这套统计量。
- 目标标准化：`(y - y_mean) / y_std`，统计量只来自训练集，`std` 用 `ddof=0` 并检查大于 0；保留原始目标与统计量，用于还原尺度。
- 网络：`nn.Sequential(Linear(10, hidden_dim), ReLU(), Linear(hidden_dim, 1))`，`hidden_dim` 默认 16，最后不接激活函数（标准化目标有正有负）。
- 两套配置：`SGD(lr=0.01, momentum=0, weight_decay=0)` 与 `Adam(lr=0.001, weight_decay=0)`，其余保持默认。
- 共同设置：CPU、`float32`、`batch_size=32`、`drop_last=False`、200 轮。
- 初始化：`torch.manual_seed(42)` 建一次初始网络，用 `copy.deepcopy` 复制出两份独立模型；两组各自新建 DataLoader，`Generator().manual_seed(42)`、`num_workers=0`，保证每轮样本顺序一致。
- 每轮更新结束后，用当时的模型重新评价**完整训练集与完整验证集**，记录标准化目标上的 MSE；验证 MSE 最小时保存该轮参数（相同保留更早一轮）。
- 训练结束后恢复各自最佳参数，按 `pred_raw = pred_scaled * y_std + y_mean` 还原，再与原始目标比较得到 MAE / RMSE。
- 基线：训练目标均值作为固定预测；`LinearRegression` 用标准化输入 + 原始目标拟合。
- 按验证 RMSE 在两组 MLP 中选择，相同则选 SGD；测试集只评价均值基线、线性回归与选定的 MLP，另一组测试列留空。

两组学习率不同（0.01 与 0.001），因此比较的是**两套具体配置**，不把差异全部归因于优化器算法本身。

## 隐藏层宽度修改检查

位置：`models.py` 的 `build_mlp(hidden_dim=...)`，宽度同时控制第一层的输出维数与第二层的输入维数。

`run_mlp.py` 末尾用 `hidden_dim=8` 另建一个网络只做前向检查：

```text
宽度 16：输入(32, 10) → Linear(32, 16) → ReLU(32, 16) → Linear(32, 1)
宽度 8 ：输入(32, 10) → Linear(32, 8)  → ReLU(32, 8)  → Linear(32, 1)
```

参数量同时从 193 降到 97（176+17 → 88+9）。只改第一层而不改第二层会导致形状无法衔接；宽度变化不影响"每条样本只有一个预测值"。该检查不重新训练、不追加图表，正式结果仍用宽度 16 的模型。

## 结果文件

| 文件 | 内容 |
|---|---|
| `results/step_trace.csv` | 实践一三行记录（清梯度后 / 反传后 / 更新后）的 `stage`、`weight`、`bias`、`weight_grad`、`bias_grad`、`loss` |
| `results/history.csv` | 两组配置各 200 行：`config`、`learning_rate`、`epoch`、`train_MSE`、`validation_MSE`、`last_batch_loss`、`target_scale`、`is_best` |
| `results/metrics.csv` | 均值基线、线性回归、SGD MLP、Adam MLP 各一行；学习率、最佳轮次、训练/验证 MAE 与 RMSE、规定模型的测试指标、`selected_mlp`；均为原始目标单位 |
| `results/loss_curves.png` | 双面板训练曲线：左 SGD、右 Adam，训练/验证 MSE，绿色虚线标出各自最佳轮次（标准化目标、线性纵轴） |
| `results/rmse_comparison.png` | 四组模型的训练/验证 RMSE 分组柱状图（原始目标单位），MLP 使用各自最佳轮次参数 |

## 本次运行结果

- 两组的最佳轮次：SGD 第 167 轮（验证 MSE 0.426 → 验证 RMSE 50.47），Adam 第 88 轮（验证 MSE 0.416 → 验证 RMSE 49.86）。两组都训练满 200 轮，但最终使用的是最佳轮次参数，而不是最后一轮。
- 按验证 RMSE 选定 **MLP + Adam**。其测试 RMSE 为 55.90，线性回归为 56.97，均值基线为 76.91（原始目标单位）。
- 均值基线的验证 RMSE 为 76.16，说明不看任何特征、直接预测平均值的误差非常大；两个 MLP 与线性回归都在 50 附近，量级相近。
- 两条训练曲线都在前 50 轮内快速下降后趋于平缓；Adam 的验证曲线在第 88 轮附近触底后略微回升，SGD 的验证曲线一直略低于训练曲线直到后期才接近。

## 解读提醒

- 标准化目标上的 MSE 与原始单位的 RMSE 不是同一个尺度，不能直接比较数值大小；曲线只用于看趋势和选轮次。
- 最后一批的 `loss` 不能当作整轮训练 MSE：最后一批只有 9 条样本，且是用该批更新前的参数算的。`history.csv` 同时保留 `last_batch_loss` 与 `train_MSE`，可直接看出两者差异（本周末尾分别相差 0.0012 与 0.0179）。
- 验证集只用于选轮次和选配置，测试集只评价一次；不合并训练与验证集重训，也不扩大参数搜索。
- 任务不要求 MLP 超过线性回归，也不要求 Adam 超过 SGD；在 442 条样本、10 个特征、265 条训练样本的量级下，线性模型本来就是很强的基线。
- ReLU 去掉后，两个 `Linear` 叠加等价于一个线性变换，网络就退化成线性回归；最后不接 ReLU 是因为标准化目标有正有负，接了会把负值一律截成 0。
