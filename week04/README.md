# 第四周：PCA 降维与 K-means 聚类

本周在 scikit-learn 内置的 diabetes 数据集上完成两项实验：PCA 前后的线性回归对照，以及 K-means 无监督聚类。数据通过 `load_diabetes(as_frame=True, scaled=False)` 读取，共 442 条样本、10 个输入特征；回归目标为 `target`。

## 文件说明

- `run_pca_clustering.py`：固定数据划分、训练集标准化、PCA 回归比较、重构误差、K-means 聚类、表格与图片导出。
- `requirements.txt`：Python 依赖版本。
- `results/`：脚本运行后生成的四份 CSV 和三张 PNG。

## 环境、安装与运行

已验证环境为 Python 3.13、numpy 2.5.3、pandas 3.0.6、scikit-learn 1.9.1、matplotlib 3.11.2。

从仓库根目录执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r week04\requirements.txt
.\.venv\Scripts\python.exe week04\run_pca_clustering.py
```

若本机已有装好依赖的 Python 3.13 环境，也可以直接用它运行最后一条命令。脚本会自动创建 `week04/results/`。

## PCA 回归设置

- 按随机种子 42，将样本固定划分为训练 265 条、验证 88 条、测试 89 条；所有模型使用同一份划分。
- `StandardScaler` 只在 10 维训练特征上 `fit`，再将同一套均值和标准差用于验证、测试特征；`target` 保持原样。
- 比较均值基线、10 维标准化特征的 `LinearRegression`、PCA 2 维 + `LinearRegression`、PCA 5 维 + `LinearRegression`。
- PCA 均使用 `svd_solver="full"`、`whiten=False`，仅在标准化训练特征上拟合；验证和测试集只调用相同 PCA 的 `transform`。
- 只依据两组 PCA 回归的验证 RMSE 选择 2 维或 5 维；如果相同则选 2 维。测试集只评价均值基线、未降维线性回归和选定的 PCA 回归，未选定 PCA 的测试列留空。
- 重构误差在标准化后的 10 维输入空间计算：`mean((X_scaled - pca.inverse_transform(pca.transform(X_scaled))) ** 2)`。它衡量输入信息还原程度，不是 `target` 的预测误差。

## K-means 设置

- 聚类输入为 `X_train_scaled`：训练集的全部 10 个标准化特征；不使用 `target`。
- 分别拟合 `KMeans(n_clusters=2, random_state=42, n_init=10)` 和 `KMeans(n_clusters=3, random_state=42, n_init=10)`。
- 用欧氏距离轮廓系数比较组内紧密和组间分离；轮廓系数不是分类准确率，组号也不是“正常”或“异常”的标签。
- 组内统计使用同一批训练样本的原始 `bmi`、`bp`；聚类图使用已经拟合的 2 维 PCA 将 10 维训练样本和 10 维聚类中心投影到 PC1、PC2 后展示。因此图是“10 维聚类，2 维展示”。

## 结果文件

| 文件 | 内容 |
|---|---|
| `results/metrics.csv` | 四组回归模型的训练、验证 MAE/RMSE；规定模型的测试 MAE/RMSE；`selected_pca=yes` 表示由验证集选定的 PCA 配置。 |
| `results/pca_summary.csv` | PCA 2、5 维的累计解释方差比例，以及训练、验证重构 MSE。 |
| `results/cluster_metrics.csv` | K=2、K=3 的轮廓系数、随机种子、`n_init` 和聚类使用的特征维数。 |
| `results/cluster_summary.csv` | 每个 K、每个组的样本数、原始 `bmi` 均值、原始 `bp` 均值。 |
| `results/rmse_comparison.png` | 四组回归模型训练与验证 RMSE 分组柱状图。 |
| `results/pca_variance.png` | 训练特征保留 1–10 个主成分时的累计解释方差，并标出 2、5 维。 |
| `results/cluster_comparison.png` | K=2、K=3 的双面板聚类图，点和中心都投影到 PC1、PC2。 |

## 本次运行结果

- 两个主成分累计解释约 54.9% 的训练输入变化，五个主成分累计解释约 83.5%。相应的训练重构 MSE 为 0.451 与 0.165，说明保留更多主成分时能更接近地恢复标准化输入。
- PCA 2 维、5 维回归的验证 RMSE 分别为 61.519、48.992，因此按照任务规则选择 PCA 5 维。其测试 RMSE 为 58.208；未降维的 10 特征线性回归测试 RMSE 为 56.970，均值基线为 76.906。这里说明 PCA 5 维能明显优于基线，但压缩仍可能损失一部分对预测有用的信息。
- K=2、K=3 的轮廓系数分别为 0.233、0.186。在本次欧氏距离和 10 维标准化输入下，K=2 的组内紧密、组间分离程度略好；这不是“正确类别”的准确率。

## 解读提醒

PCA 的主成分是原始特征的线性组合，不是直接挑出两列。累计解释方差描述保留的输入变化比例，不能直接当作预测准确率；保留更多输入方差也不保证验证 RMSE 必然更低。K-means 的聚类结果只说明给定特征和距离定义下的相似分组，不能单凭组号赋予业务或医学含义。
