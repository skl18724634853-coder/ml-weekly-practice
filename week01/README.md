# 第一周：diabetes 数据探索

使用 `load_diabetes(as_frame=True, scaled=False)` 加载 scikit-learn 内置数据：442 个样本、10 个特征，目标为一年后病情进展的定量指标。查看结构、缺失值、描述统计和相关性，并绘图；本周不训练模型。行号不代表时间。

## 文件与环境

- `diabetes_explore.py`：读取数据、打印统计、生成直方图和特征散点图。
- `requirements.txt`：精确依赖版本；Python 3.13。
- `outputs/`：运行产生的图表。

## 安装与运行（Windows PowerShell）

从仓库根目录执行。直接使用虚拟环境解释器，无需激活脚本。

```powershell
cd week01
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe diabetes_explore.py
.\.venv\Scripts\python.exe diabetes_explore.py --feature s3
```

使用其他 Python 3.13 安装时，可将 `py -3.13` 替换为该解释器路径。macOS/Linux 使用 `python3.13 -m venv .venv`，后续使用 `.venv/bin/python`。

## 结果

| 文件 | 内容 |
|---|---|
| `outputs/01_target_distribution.png` | 目标分布直方图及均值参考线 |
| `outputs/02_scatter_bmi.png` | bmi 与目标的散点图 |
| `outputs/02_scatter_s3.png` | 改用 s3 后的散点图 |

默认运行生成前两张图；第二条运行命令补充 s3 图，并重新生成同一张目标直方图。
数据没有缺失值。bmi 与目标的相关系数约 0.586，s3 约 -0.395。替换特征可比较关系方向，但相关性不能证明因果关系，也不能据此判断预测模型的泛化效果。
