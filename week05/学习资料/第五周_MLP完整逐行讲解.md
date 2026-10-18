# 第五周 MLP 与 PyTorch 训练基础：一份从零开始的完整逐行讲义

> **这就是本项目唯一的代码讲解文件。** 建议把它和相应的 `.py` 源文件并排打开：每读一段，就去源文件找到相同的行号。看不懂时不要硬背，先回来看“名词字典”和“五步训练流程”。

---

## 先说任务目标：这份作业到底在练什么

本周不是要求你背会某个模型名字，而是要把今后所有深度学习反复出现的训练流程真正看懂：

```text
准备数据
  → 模型根据当前参数作预测（前向传播）
  → 用正确答案衡量错得多不多（损失）
  → 算出每个参数该往哪边改（反向传播）
  → 真正更新参数（优化器）
  → 用没参与改参数的数据检查效果（验证 / 测试）
```

这份作业有两部分。

1. **实践一**只使用两条样本和一个 `Linear(1,1)`。先让权重、偏置都等于 0，再亲眼观察：清梯度、前向、MSE、反向传播、SGD 更新各自到底干什么。它的结果必须能手算核对：`dw=-10`、`db=-6`，更新后 `w=1`、`b=0.6`，损失由 `10` 降到 `1.06`。
2. **实践二**使用 diabetes 回归数据集。它有 442 条样本，每条 10 个输入特征，要预测 1 个连续数值。程序按 60% / 20% / 20% 分训练、验证、测试集；训练 `10→16→1` 的 MLP；比较 SGD 和 Adam；保留验证集最好的模型；最终输出 CSV 和图表。

本项目的文件并不是八份互不相干的作业，而是一条流水线上的不同岗位：

```text
data.py        负责把原始资料整理成模型能吃的数据
models.py      负责规定模型长什么样
optimizers.py  负责规定参数怎么更新
training.py    负责反复训练
evaluation.py  负责只预测、只评分，绝不更新
reporting.py   负责保存 CSV、绘制 PNG
run_step.py    实践一的总指挥
run_mlp.py     实践二的总指挥
```

把它想成饭店：`data.py` 洗菜，`models.py` 决定菜谱，`training.py` 炒菜，`evaluation.py` 试菜评分，`reporting.py` 写菜单和拍照，两个 `run_*.py` 则是总厨安排先后顺序。

---

## 所有特殊名词：先用人话说清楚

| 名词 / 写法 | 最简单的解释 | 在本作业中具体做什么 |
|---|---|---|
| Python | 让人把计算步骤写给电脑的语言 | 所有 `.py` 文件由它执行 |
| PyTorch / `torch` | 深度学习工具箱 | 建张量、模型、自动求导、更新参数 |
| `numpy` / `np` | 处理数字数组的工具箱 | 算均值、标准差、指标 |
| `sklearn` | 传统机器学习工具箱 | 读数据、划分、标准化、线性回归 |
| 样本 sample | 一条完整记录 | diabetes 中的一位病人记录 |
| 特征 feature | 描述样本的一个数字 | 每位病人的 10 个输入数 |
| 目标 / 标签 target / `y` | 希望模型猜对的标准答案 | 一个连续数值 |
| 输入 / `x` | 交给模型的题目 | 每条样本的 10 个特征 |
| 张量 tensor | PyTorch 认识的多维数字盒子 | `x`、`y`、参数和预测都装在里面 |
| 形状 shape | 数字盒子每一层有多少格 | `(32,10)` 是 32 条样本、每条 10 个数 |
| `float32` | 32 位小数格式 | 本周网络的统一计算格式 |
| 模型 model | 根据输入产生预测的可调机器 | 线性模型或 MLP |
| 参数 parameter | 模型内部可被训练改变的数字旋钮 | `weight` 与 `bias` |
| 权重 weight | 决定输入影响有多大 | 线性层主要的可学习数字 |
| 偏置 bias | 输入全为 0 时仍可产生的基础偏移 | 线性层额外的可学习数字 |
| `nn.Linear` | 加权相加并加偏置的层 | 完成 `W×x+b` |
| MLP | 多层感知机 | `Linear → ReLU → Linear` 的网络 |
| 隐藏层 | 输入与输出之间的中间加工层 | 正式模型宽度为 16 |
| ReLU | 负数变 0，正数保留 | 给网络加入非线性能力 |
| 前向传播 forward | 用当前参数从输入算到预测 | `prediction = model(x)` |
| 损失 loss | 把“错多少”压成一个分数 | 本周用 MSE，越小越好 |
| MSE | 每个误差平方再平均 | 训练时用的损失 |
| MAE | 绝对误差平均 | 最终评价中的“平均差多少” |
| RMSE | MSE 开根号 | 单位回到原目标，易理解 |
| 梯度 gradient / `.grad` | 参数应朝哪边改、改多少才会让损失变小的建议 | `backward()` 算出它 |
| 反向传播 backward | 从损失倒着计算所有参数的梯度 | `loss.backward()` |
| 优化器 optimizer | 根据梯度真正改参数的执行者 | SGD 或 Adam |
| 学习率 `lr` | 每一次实际改动走多大一步 | 大了易乱跳，小了学得慢 |
| SGD | 主要跟随当前梯度走一步 | 两组实验之一 |
| Adam | 会参考近期梯度方向和波动调节步长 | 另一组实验 |
| 批次 batch | 一次交给模型的一小撮样本 | 每批最多 32 条 |
| DataLoader | 自动按批给数据的送货员 | 送训练集给训练循环 |
| epoch（轮） | 训练集完整走完一遍 | 每种配置跑 200 轮 |
| 训练 / 验证 / 测试集 | 分别负责学参数、选配置、最后总评 | 三者不能混用 |
| 标准化 | 减均值、除标准差，使数字大致围绕 0 | 让训练更稳定 |
| 数据泄漏 | 偷看验证/测试信息再训练 | 通过“只在训练集 fit”避免 |
| 基线 baseline | 简单参照物 | 检查复杂模型是否真学到东西 |
| `state_dict` | 模型参数的快照照片 | 保存验证集最好一轮 |
| `model.train()` | 告诉模型现在训练 | 训练批次前调用 |
| `model.eval()` | 告诉模型现在考试 / 评价 | 预测前调用 |
| `torch.no_grad()` | 暂时不记录求导 | 初始化和评价时省资源 |
| CSV / PNG | 表格文件 / 图片文件 | 保存历史、指标、曲线、柱状图 |

---

## 总体思路：先记住五句核心代码

无论是实践一的两条样本，还是实践二的 265 条训练样本，真正改变模型参数的核心总是下面五句：

```python
optimizer.zero_grad()
prediction = model(xb)
loss = loss_fn(prediction, yb)
loss.backward()
optimizer.step()
```

用“练投篮”理解：先擦掉上一轮的改正意见；按当前动作投一次；量一量错了多少；分析每个关节该怎样调；最后真的把动作调掉。请记住：**`backward()` 只写出调整建议，`step()` 才实际改变参数。**

实践二的完整路线是：

```text
读取 diabetes
→ 固定种子切分训练 / 验证 / 测试
→ 只用训练集统计量标准化
→ DataLoader 按批送数据
→ 建一份初始 MLP，再复制给 SGD / Adam
→ 每批执行五步训练
→ 每轮在完整训练集、完整验证集上重新评分
→ 保存验证集最佳 state_dict
→ 恢复最佳参数，换回原始单位算 MAE / RMSE
→ 写 CSV、画 PNG
```

---

## 读代码时反复遇到的符号

| 写法 | 作用 | 特别提醒 |
|---|---|---|
| `import` | 把工具箱拿进来 | 导入不等于开始训练 |
| `from ... import ...` | 从大工具箱取一个抽屉 | `from torch import nn` 后可写 `nn.Linear` |
| `def` | 定义可重复用的小步骤（函数） | 定义本身不会立刻运行 |
| `class` | 定义一种固定格式的记录本 | `UpdateTrace`、`BestModel` 是记录本 |
| `return` | 把结果交回调用处 | 函数本次调用到这里结束 |
| `=` | 把右边结果放进左边名称 | 不等于数学里的恒等 |
| `==` | 比较是否相等 | 判断使用它，赋值不用它 |
| `:` 与缩进 | 开始并归属一个代码块 | Python 用缩进代替花括号 |
| `()` | 调用函数或组合表达式 | `model(x)` 是让模型计算 |
| `[]` | 列表、按位置取值、读字典字段 | `rows[0]`、`row["loss"]` |
| `{}` | 字典，给数据贴名称 | `{"epoch": 1}` |
| `.` | 取对象的属性或功能 | `loss.backward()` |
| `None` | 暂时没有值 | 反传前 `.grad` 是它 |
| `True` / `False` | 真 / 假 | `shuffle=True` 代表打乱 |
| `f"...{变量}..."` | 把变量值插进文字 | 用于打印和报错 |
| `-> 类型`、`list[float]`、`A | B` | 类型提示 | 只给人和编辑器看，不改计算 |
| `assert` | 条件不成立就立刻报错 | 防止程序悄悄算错 |

下面开始按实际调用顺序，对所有可执行 Python 文件逐行解释。每段都会说明这一行从哪里拿东西、做了什么、交给谁，以及关键单词的作用。

---

# 第一部分：用两个样本看懂一次梯度更新

本部分只讲实践一。它的目的不是训练一个有用的模型，而是把你第二周手写过的 `dw`、`db`，对应到 PyTorch 的 `backward()` 和 `step()` 上。

运行入口是 `run_step.py`。它会依次调用：

```text
run_step.py
  ├─ data.make_step_tensors()        取出两条样本
  ├─ models.build_zeroed_linear()    造一个权重、偏置都为 0 的直线模型
  ├─ optimizers.build_optimizer()    造 SGD 更新器
  ├─ training.one_update_trace()     完成一次“清梯度→前向→损失→反传→更新”
  └─ reporting.save_step_trace()     保存三行结果
```

本例固定的数据是：

```text
x = [[1.0], [2.0]]
y = [[2.0], [4.0]]
```

模型是一条直线 `pred = w × x + b`。开始时 `w=0`、`b=0`，学习率为 `0.1`。

## 1. `data.py`：第 39–47 行，造出两条样本

原代码：

```python
39  def make_step_tensors() -> tuple[torch.Tensor, torch.Tensor]:
40      """实践一的样本说明。"""
45      x = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
46      y = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
47      return x, y
```

- **第 39 行**：`def` 的意思是“定义一个步骤”。`make_step_tensors` 是这个步骤的名字，可理解成“制作实践一张量”。圆括号里没有内容，表示调用它时不需要你额外提供材料。`->` 后面的 `tuple[torch.Tensor, torch.Tensor]` 是类型说明：它会交回一对 PyTorch 张量；第一个装输入，第二个装标签。它是说明书，不是实际计算。
- **第 40–44 行**：三引号中的文字是函数说明。Python 不会把这些字当作训练指令执行。
- **第 45 行**：左边的 `x` 是我们给输入数据起的名字。`=` 是“把右边算出的东西放进左边的盒子”，不是数学课里永远相等的意思。`torch` 是 PyTorch 工具箱；`.` 表示从这个工具箱拿 `tensor` 这个功能；`tensor(...)` 把普通数字变成神经网络可计算的张量。`[[1.0], [2.0]]` 有两层方括号：外层表示有两条样本，内层表示每条样本只有一个特征。`dtype=` 是“指定数据类型”；`torch.float32` 是 32 位小数，是本周统一使用的计算格式。
- **第 46 行**：含义与第 45 行相同，不过 `y` 装的是正确答案。它也是 `(2, 1)`：两行样本，每行一个目标值。让 `x` 与 `y` 行数相同，才能保证第 1 个输入配第 1 个答案。
- **第 47 行**：`return` 是“把结果交回去”。逗号把 `x` 和 `y` 打包成一对。调用处可以用 `x, y = ...` 同时接住它们。

**老人式记忆法：**`x` 是两道题，`y` 是两道题的答案；张量的第一维永远先数“有几道题”。

## 2. `models.py`：第 32–43 行，造一台最简单的预测机器

原代码：

```python
32  def build_zeroed_linear() -> nn.Linear:
39      model = nn.Linear(1, 1)
40      with torch.no_grad():
41          model.weight.zero_()
42          model.bias.zero_()
43      return model
```

- **第 32 行**：函数名 `build_zeroed_linear` 直译是“建立一个已清零的线性模型”。`nn.Linear` 是这个函数将交回的模型类型。
- **第 39 行**：`nn` 是 PyTorch 里“神经网络工具”的简称。`Linear(1, 1)` 的第一个 `1` 表示每条样本有 1 个输入特征，第二个 `1` 表示每条样本产出 1 个预测值。它内部保存两个可学习参数：权重 `weight` 和偏置 `bias`。
- **第 40 行**：`with` 可以理解为“在下面缩进的这块范围内，临时采用某种规则”。`torch.no_grad()` 的规则是“不建立求导记录”。这里我们只是手动设初值，不是训练，也不希望 PyTorch 把设零误解成一次需要反传的计算。
- **第 41 行**：`model.weight` 取出模型的权重。`zero_()` 末尾的下划线非常重要：它表示“直接原地改这个东西”，不是造一个新的零张量。执行后 `w=0`。
- **第 42 行**：同理，直接把偏置 `b` 改为 `0`。
- **第 43 行**：把已经设好初值的模型交还。

为什么不在真正的 MLP 中也把所有权重设为 0？因为多个隐藏神经元如果完全一样，就会一直学成一样，失去分工。本例只含一个输入和一个输出，置零仅仅是为了让手算数字清楚。

## 3. `optimizers.py`：第 39–83 行，决定怎样改参数

原代码：

```python
39  STEP_LEARNING_RATE = 0.1
44  def build_optimizer(optimizer_name, model, learning_rate,
45                      momentum=0.0, weight_decay=0.0):
69      if optimizer_name == "sgd":
70          return torch.optim.SGD(
71              model.parameters(),
72              lr=learning_rate,
73              momentum=momentum,
74              weight_decay=weight_decay,
75          )
76      if optimizer_name == "adam":
78          return torch.optim.Adam(
79              model.parameters(),
80              lr=learning_rate,
81              weight_decay=weight_decay,
82          )
83      raise ValueError(...)
```

- **第 39 行**：全大写的 `STEP_LEARNING_RATE` 是“固定配置”的惯例写法，告诉读者这不是训练中随时变化的普通变量。`0.1` 是学习率：每次听取梯度建议后，实际走多大步。
- **第 44–50 行**：函数接收五样东西。`optimizer_name` 是字符串，例如 `"sgd"`；`model` 是要训练的模型；`learning_rate` 是学习率；后面两个带 `=0.0` 的参数是默认值，调用时没写就使用 0。
- **第 69 行**：`if` 是“如果”。`==` 是“比较是否相同”，不能写成一个 `=`。如果名字正好是小写的 `"sgd"`，就走下面的分支。
- **第 70 行**：`torch.optim.SGD` 造出 SGD 优化器。注意它没有计算梯度的功能；它只会在稍后的 `step()` 中读取梯度并更新参数。
- **第 71 行**：`model.parameters()` 拿到模型中全部可学习参数。本例就是权重和偏置。圆括号表示这是一次函数调用。
- **第 72 行**：`lr` 是 learning rate 的缩写；`lr=learning_rate` 表示把函数收到的学习率交给 SGD。
- **第 73–74 行**：`momentum=0` 表示不使用动量；`weight_decay=0` 表示不额外做 L2 惩罚。这正是任务指定的最朴素 SGD。
- **第 76–82 行**：如果名字是 `"adam"`，就造 Adam。实践一不会走这个分支，但实践二会使用。Adam 的 `betas`、`eps` 等没有写出来，表示使用 PyTorch 的默认设置。
- **第 83 行**：`raise ValueError` 是主动报错。如果有人错把名字写成 `"SGD"` 或 `"adma"`，程序不会悄悄选错算法，而是立即告诉你输入无效。

## 4. `training.py`：第 28–119 行，真正完成一次更新

### 4.1 第 28–42 行：`UpdateTrace`，给三次观察记账

```python
28  @dataclass
29  class UpdateTrace:
38      rows: list[dict[str, float | str | None]] = field(default_factory=list)
39      initial_prediction: np.ndarray | None = None
40      updated_prediction: np.ndarray | None = None
41      loss_before: float | None = None
42      loss_after: float | None = None
```

- **第 28 行**：`@dataclass` 是一个“简化记账本”的标记。它会自动帮我们生成保存这些字段的基础功能。
- **第 29 行**：`class` 定义一种记录本的样式；`UpdateTrace` 是这种记录本的名字。
- **第 38 行**：`rows` 是列表，里面会装三个字典。每个字典是一行记录，字段名如 `"weight"`、`"loss"`。`float | str | None` 的竖线表示一个格子允许放小数、文字或“暂时没有”。`field(default_factory=list)` 的意思是：每新建一本记录本，就新给它一个空列表；不能直接写 `[]`，否则不同记录本可能共用同一个列表。
- **第 39–42 行**：四个字段初始都是 `None`。`initial_prediction` 记录更新前的预测；`updated_prediction` 记录更新后重新算的预测；两个 `loss` 对应更新前后。竖线类型说明只是人和编辑器的提示。

### 4.2 第 45–47 行：把张量里的单个数取出来

```python
45  def _pairs(weight: torch.Tensor, bias: torch.Tensor) -> tuple[float, float]:
47      return float(weight.item()), float(bias.item())
```

- 函数名前的下划线 `_pairs` 是惯例：它是本文件内部的小助手，不希望其他文件把它当公开接口。
- 参数名 `weight`、`bias` 分别接收权重张量和偏置张量。
- `item()` 把“只有一个数的 PyTorch 张量”取成普通 Python 数；外层 `float(...)` 再确保它是小数。逗号表示交回一对数。

### 4.3 第 50–62 行：一次更新的函数外壳

```python
50  def one_update_trace(model, x, y, loss_fn, optimizer) -> UpdateTrace:
57      """按清梯度→前向→损失→反传→更新走一次。"""
```

- `model` 是线性模型；`x`、`y` 是两条输入与答案；`loss_fn` 是 MSE 损失函数；`optimizer` 是 SGD。
- 函数最终交回 `UpdateTrace`，而不是只交回一个损失。这是为了让你能观察训练的每个阶段。

### 4.4 第 64–76 行：内部小函数 `record`

```python
64  def record(stage: str, loss: float | None) -> dict[str, float | str | None]:
66      weight, bias = _pairs(model.weight, model.bias)
67      weight_grad = None if model.weight.grad is None else float(model.weight.grad.item())
68      bias_grad = None if model.bias.grad is None else float(model.bias.grad.item())
69      return {
70          "stage": stage,
71          "weight": weight,
72          "bias": bias,
73          "weight_grad": weight_grad,
74          "bias_grad": bias_grad,
75          "loss": loss,
76      }
```

- **第 64 行**：`record` 是嵌套在 `one_update_trace` 内的小函数，只供这一次训练使用。它接收阶段名和可能存在的损失。
- **第 66 行**：先从当前模型拿权重、偏置，交给 `_pairs` 转成普通数。左侧的两个名字用逗号同时接住返回的两个数。
- **第 67 行**：这是 Python 的“三元表达式”，读法是：“如果 `model.weight.grad is None`，就取 `None`；否则取梯度里的一个数并转为小数。”`is None` 是专门问“是不是没有值”的可靠写法。清梯度后还没运行反传，因此这里确实没有梯度。
- **第 68 行**：偏置梯度按同样办法处理。
- **第 69–76 行**：花括号建立字典；每个 `"名字": 值` 都是一个贴好标签的格子。这一行记录能直接写进 CSV。

### 4.5 第 78–119 行：五个核心步骤，逐步发生了什么

```python
78  trace = UpdateTrace()
81  optimizer.zero_grad()
82  trace.rows.append(record("after_zero_grad", None))
85  prediction = model(x)
87  assert prediction.shape == y.shape, ...
90  loss = loss_fn(prediction, y)
91  trace.loss_before = float(loss.item())
95  loss.backward()
96  trace.rows.append(record("after_backward", trace.loss_before))
100 optimizer.step()
101 weight_after, bias_after = _pairs(model.weight, model.bias)
104 with torch.no_grad():
105     updated_prediction = model(x)
106     loss_after = float(loss_fn(updated_prediction, y).item())
108 trace.rows.append(record("after_step", loss_after))
109 trace.initial_prediction = prediction.detach().numpy().copy()
110 trace.updated_prediction = updated_prediction.detach().numpy().copy()
111 trace.loss_after = loss_after
114 learning_rate = float(optimizer.param_groups[0]["lr"])
116 assert abs(weight_after - (...)) < 1e-6
117 assert abs(bias_after - (...)) < 1e-6
119 return trace
```

- **第 78 行**：`UpdateTrace()` 是“按 UpdateTrace 样式新建一本空账本”。结果存进 `trace`。
- **第 81 行**：`optimizer.zero_grad()` 把已有梯度清空。PyTorch 默认会把多次 `backward()` 的梯度相加，因此每一个新批次开始时必须清理。实践一虽是第一次训练，仍保留这行正确流程。
- **第 82 行**：`append` 的意思是“往列表尾部添一项”。此时 `record` 读到权重 `0`、偏置 `0`、两个梯度 `None`、损失 `None`，得到第一行记录。
- **第 85 行**：`model(x)` 是前向计算。模型把两个输入带入 `pred=w×x+b`。此刻 `w=b=0`，所以预测是 `[[0],[0]]`。它只计算预测，还没有学习。
- **第 87 行**：`assert` 是“断言检查”。如果后面的条件为假，立即报错。`.shape` 看张量尺寸；这里确保预测 `(2,1)` 和标签 `(2,1)` 完全一致，防止样本错配或广播错误。逗号后面的 f 字符串是报错时给人的提示。
- **第 90 行**：`loss_fn(prediction, y)` 计算均方误差：`((0-2)²+(0-4)²)/2=10`。MSELoss 会输出只有一个数的张量。
- **第 91 行**：`loss.item()` 取出这个单数，`float` 转成普通小数，保存为更新前的损失。
- **第 95 行**：最关键。`backward()` 从损失沿计算图反向求导，把 `d loss / d weight` 写到 `model.weight.grad`，把 `d loss / d bias` 写到 `model.bias.grad`。这里结果分别是 `-10` 和 `-6`。它**没有**改变权重和偏置。
- **第 96 行**：把“反传后”的状态记成第二行；此时参数仍是 `0,0`，但梯度已经出现。
- **第 100 行**：第二个关键。`step()` 读取两个 `.grad`，按 SGD 规则更新：`w = 0 - 0.1×(-10)=1`，`b = 0 - 0.1×(-6)=0.6`。它**没有**重算梯度。
- **第 101 行**：马上读取更新后的权重、偏置，留给下面核对。
- **第 104–106 行**：再次进入“不要记录梯度”的范围，重新调用 `model(x)`。这是必要的：第 85 行保存的 `prediction` 是旧参数算出的旧结果，不会因为参数变了而自动变化。重新前向得到 `[1.6,2.6]`，再得到新损失 `1.06`。
- **第 108 行**：把更新后的参数、仍保留的旧梯度、以及重新算的新损失记成第三行。
- **第 109–111 行**：`detach()` 把张量从求导计算图中拆下来；`.numpy()` 转成 NumPy 数组；`.copy()` 再复制一份，以后不随原变量改变。这样输出时能同时展示新旧预测。
- **第 114 行**：优化器的 `param_groups` 是配置列表；`[0]` 取第一组参数；`["lr"]` 取其中名为 `lr` 的学习率。这里读出的就是 `0.1`。
- **第 116–117 行**：`abs(...)` 是绝对值；`< 1e-6` 表示误差小于一百万分之一就算通过。两句 `assert` 验证 SGD 实际做的事，确实等于“旧参数减学习率乘梯度”。
- **第 119 行**：把完整的三阶段账本交回 `run_step.py`。

## 5. `run_step.py`：入口怎样把所有部件接起来

### 第 13–24 行：导入

```python
13  from __future__ import annotations
15  import sys
17  import torch
18  from torch import nn
20  import data
21  import models
22  import optimizers
23  import reporting
24  import training
```

- **第 13 行**：让类型提示可以更灵活地引用尚未完全定义的类型；对本周计算逻辑没有影响。
- **第 15 行**：`sys` 是 Python 自带的系统工具，此处只用来调整终端中文编码。
- **第 17–18 行**：导入 PyTorch 与其中的神经网络抽屉 `nn`。
- **第 20–24 行**：导入本周五个组件。单独 `import` 后，必须写 `data.make_step_tensors()` 这样清楚地说明功能属于哪个文件，组会时也容易追溯。

### 第 27–33 行：保证中文能正常显示

```python
27  def use_utf8_output() -> None:
29      for stream in (sys.stdout, sys.stderr):
30          try:
31              stream.reconfigure(encoding="utf-8")
32          except (AttributeError, ValueError):
33              pass
```

- `None` 表示这个函数只做设置，不交回数据。
- `sys.stdout` 是正常输出，`sys.stderr` 是错误输出；圆括号中的逗号把它们组成一对。
- `for` 表示“依次对每一个”。`try` 的意思是“尝试执行”；如果某种终端没有 `reconfigure`，`except` 接住两种常见错误；`pass` 表示“什么也不做，继续运行”。这部分和 MLP 无关，只是防止 Windows 中文乱码。

### 第 36–58 行：把输出排成容易读的表

这里的 `format_prediction` 用 `reshape(-1)` 将 `(2,1)` 展平为两个数，`join` 用逗号拼成文字；`print_trace_table` 用 f 字符串规定每列宽度。它们只影响显示，**不参与损失、反传或参数更新**。这正是把“展示”与“训练”分开的例子。

### 第 61–139 行：`main()` 的执行顺序

```python
61  def main() -> None:
62      use_utf8_output()
65      x, y = data.make_step_tensors()
76      model = models.build_zeroed_linear()
77      optimizer = optimizers.build_optimizer("sgd", model, ...)
81      loss_fn = nn.MSELoss()
84      with torch.no_grad():
85          initial_prediction = model(x)
86          initial_mse = float(loss_fn(initial_prediction, y).item())
89      assert abs(initial_mse - 10.0) < 1e-6, ...
93      trace = training.one_update_trace(model, x, y, loss_fn, optimizer)
95      print_trace_table(trace.rows)
98      after_backward = trace.rows[1]
99      after_step = trace.rows[2]
124     csv_path = reporting.save_step_trace(trace.rows)
138 if __name__ == "__main__":
139     main()
```

- **第 62 行**：先修正终端文字显示。
- **第 65 行**：调用数据组件，逗号左边一次接住 `x`、`y`。
- **第 76 行**：调用模型组件，取得权重偏置均为零的 `nn.Linear(1,1)`。
- **第 77–80 行**：调用优化器组件，明确传入 `"sgd"`、模型、学习率、`momentum=0`、`weight_decay=0`。
- **第 81 行**：`nn.MSELoss()` 造出均方误差规则。变量叫 `loss_fn`，末尾的 `fn` 是 function 的简写。
- **第 84–86 行**：在不求梯度的状态下，先检查初始预测和初始 MSE。不能把这段当作训练，因为没有 `backward()` 或 `step()`。
- **第 89 行**：确认初始 MSE 必须是 10；若数据、模型或损失函数被意外改坏，立刻停止。
- **第 93 行**：把五步训练完整交给训练组件。注意 `model` 是同一个对象，函数内部更新它以后，入口中的模型也已经更新。
- **第 95 行**：把三行账本打印出来。第 98–120 行只是取第二、三行做手算说明，便于你核对 `-10`、`-6`、`1`、`0.6`、`1.06`。
- **第 124 行**：调用报告组件写出 `results/step_trace.csv`；这里保存的是已经准备好的 `trace.rows`，保存函数没有重新训练模型。
- **第 138–139 行**：这是 Python 常用入口开关。直接执行 `python run_step.py` 时，`__name__` 的值是 `"__main__"`，所以会调用 `main()`；若其他文件只是导入它，`main()` 不会自动跑，避免“导入一个文件就开始训练”。

## 6. 这部分必须能口头说出的结论

1. `loss.backward()` 对应手写的 `dw`、`db`：它只把梯度写进 `.grad`。
2. `optimizer.step()` 对应手写的 `w -= lr × dw`：它才真正改权重和偏置。
3. 每次新批次前要 `zero_grad()`，因为 PyTorch 默认累积梯度。
4. 参数更新后要重新做前向计算；旧的 `prediction` 不会自动重算。
5. 本例的数值顺序是：初始 MSE `10` → 梯度 `dw=-10, db=-6` → 新参数 `w=1, b=0.6` → 新 MSE `1.06`。

---

# 第二部分：数据、张量形状与 MLP 逐行讲解

这一部分对应 `data.py` 与 `models.py`。先记住两个完全不同的数字：

- **样本数 N**：有多少个人、多少条记录。这里训练集是 265 条、验证集 88 条、测试集 89 条。
- **特征数 10**：每个人的十项已知信息。每条样本都有 10 个特征。

所以训练特征的形状是 `(265, 10)`，不是“265 个特征”。它读作：**265 行样本，每行 10 个数。**

## 1. `data.py` 顶部：导入与固定设置（第 16–32 行）

```python
16  from __future__ import annotations
18  from dataclasses import dataclass
20  import numpy as np
21  import torch
22  from sklearn.datasets import load_diabetes
23  from sklearn.model_selection import train_test_split
24  from sklearn.preprocessing import StandardScaler
25  from torch.utils.data import DataLoader, TensorDataset
28  SEED = 42
29  BATCH_SIZE = 32
30  TRAIN_RATIO = 0.6
31  VALIDATION_RATIO = 0.2
32  TEST_RATIO = 0.2
```

- **第 16 行**：只影响类型提示的解释方式，不改模型结果。
- **第 18 行**：`dataclass` 是制作“带固定格子的记录盒”的工具，后面用它造 `DiabetesData`。
- **第 20 行**：`numpy as np` 表示导入 NumPy 并简称为 `np`。以后看到 `np.mean`，就是 NumPy 的平均数功能。
- **第 21 行**：导入 PyTorch；张量、神经网络计算都来自它。
- **第 22–24 行**：从 scikit-learn 分别拿数据集、划分工具、标准化工具。
- **第 25 行**：`TensorDataset` 把特征和标签绑成一对；`DataLoader` 再把它们切成一小批一小批。
- **第 28 行**：随机种子 `42`。随机划分、随机初始化、随机打乱如果使用同一固定种子，重复运行能得到同样的顺序和结果，便于核对。
- **第 29 行**：每批 `32` 条样本。265 除以 32 会得到 8 个完整批次，再加一个 9 条样本的小批次。
- **第 30–32 行**：三种数据占比。任务实际使用 `VALIDATION_RATIO` 与 `TEST_RATIO`；`TRAIN_RATIO` 保留在这里是为了把实验规则写清楚，60% 由“先留 40%”间接实现。

## 2. `DiabetesData`：把同一份实验数据放进一个总文件夹（第 53–88 行）

```python
53  @dataclass
54  class DiabetesData:
70      x_train_scaled: np.ndarray
73      x_train_tensor: torch.Tensor
76      y_train_scaled: torch.Tensor
79      y_train_raw: np.ndarray
82      y_mean: float
83      y_std: float
84      feature_names: list[str]
86      @property
87      def n_features(self) -> int:
88          return len(self.feature_names)
```

这些不是“正在执行的训练代码”，而是在声明这个总文件夹里将有哪些格子。

- **第 53 行**：让 Python 自动生成“把各格子装起来”的基础功能。
- **第 54 行**：`class DiabetesData` 定义一个数据总包。以后 `dataset.x_train_tensor` 的意思是“从这个总包拿训练特征张量”。
- **第 70–72 行**：`x_train_scaled`、`x_validation_scaled`、`x_test_scaled` 是 NumPy 数组，给 scikit-learn 的均值基线和线性回归用。`scaled` 表示已经做过输入标准化。
- **第 73–75 行**：同一份输入转为 PyTorch 张量，给神经网络用。三者形状分别是 `(265,10)`、`(88,10)`、`(89,10)`。
- **第 76–78 行**：标准化后的标签张量。标签特意做成 `(N,1)`，而不是 `(N,)`，以便与网络最后输出 `(N,1)` 对齐。
- **第 79–81 行**：原始单位的标签。最后的 MAE、RMSE 要面对人解释，因此要用这些原始数。
- **第 82–83 行**：只从训练集算出的标签平均值和标准差。它们相当于“尺子的零点”和“一个刻度有多长”。
- **第 84 行**：十个特征的名称列表；`list[str]` 表示“由文字组成的列表”。
- **第 86 行**：`@property` 让下面的函数可以像普通字段一样使用，即写 `dataset.n_features`，不必写小括号。
- **第 87–88 行**：`len(...)` 数列表长度，交回特征数 10。

## 3. `split_indices`：先决定谁属于训练、验证、测试（第 91–104 行）

```python
91  def split_indices(sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
96      indices = np.arange(sample_count)
97      train_idx, remaining_idx = train_test_split(
98          indices, test_size=VALIDATION_RATIO + TEST_RATIO, random_state=SEED
99      )
100     validation_idx, test_idx = train_test_split(
101         remaining_idx, test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
102         random_state=SEED,
103     )
104     return train_idx, validation_idx, test_idx
```

- **第 91 行**：函数收到总样本数 `sample_count`，将交回三个 NumPy 下标数组。
- **第 96 行**：`np.arange(442)` 会做出 `0,1,2,...,441`。这些数是每一行样本的编号，不是特征值。
- **第 97–99 行**：第一次随机划分。`test_size=0.2+0.2` 先从全部数据拿走 40%，剩下约 60% 就是训练集。`random_state=42` 固定这次抽签结果。
- **第 100–103 行**：把刚才暂存的 40% 再平分。`0.2/(0.2+0.2)=0.5`，因此其中一半验证、一半测试。
- **第 104 行**：按训练、验证、测试的顺序交回三份行号。之后所有模型使用同一批行号，比较才公平。

**为什么不能先对全体数据标准化再划分？**因为验证、测试的数据会偷偷参与均值和标准差的计算，相当于考试前偷看了考卷；这叫数据泄漏。

## 4. `prepare_diabetes`：把原始数据加工成网络能吃的数据（第 107–162 行）

### 4.1 第 112–122 行：读数据并按下标取三份

```python
112 dataset = load_diabetes(as_frame=True, scaled=False)
113 x_frame = dataset.data
114 y_array = dataset.target.to_numpy(dtype=float)
116 train_idx, validation_idx, test_idx = split_indices(len(x_frame))
117 x_train = x_frame.iloc[train_idx]
118 x_validation = x_frame.iloc[validation_idx]
119 x_test = x_frame.iloc[test_idx]
120 y_train_raw = y_array[train_idx]
121 y_validation_raw = y_array[validation_idx]
122 y_test_raw = y_array[test_idx]
```

- **第 112 行**：`load_diabetes` 读入 sklearn 自带的 diabetes 回归数据。`as_frame=True` 让输入保留为带列名的表；`scaled=False` 表示取得未预先缩放的原始输入，确保本周的 `StandardScaler` 真正由我们控制。
- **第 113 行**：`dataset.data` 取出 10 列输入表，命名为 `x_frame`。`frame` 提醒我们它是表格。
- **第 114 行**：`dataset.target` 取回归目标；`.to_numpy(dtype=float)` 把表格列转成普通小数数组。这里不先加第二个维度，因为 sklearn 基线更习惯一维标签。
- **第 116 行**：`len(x_frame)` 是样本总数 442；用前节函数得到三组行号。
- **第 117–119 行**：`.iloc[...]` 按“行位置”取输入表的训练、验证、测试部分。
- **第 120–122 行**：用同一组下标从目标数组取答案。注意：输入与答案用相同下标，这正是不会配错样本的原因。

### 4.2 第 124–128 行：输入标准化

```python
125 scaler = StandardScaler()
126 x_train_scaled = scaler.fit_transform(x_train)
127 x_validation_scaled = scaler.transform(x_validation)
128 x_test_scaled = scaler.transform(x_test)
```

- **第 125 行**：造一把 `scaler` 尺子。刚造出来时它还不知道任何数据的均值、标准差。
- **第 126 行**：`fit_transform` 包含两步：先 `fit`，只观察训练集每个特征列的平均值、标准差；再 `transform`，把训练集变为大致均值 0、标准差 1 的数。
- **第 127–128 行**：验证、测试只能调用 `transform`，也就是沿用训练集那把尺子。不能写 `fit_transform`，否则就用到了验证或测试自己的信息。

### 4.3 第 130–137 行：目标也标准化

```python
131 y_mean = float(y_train_raw.mean())
132 y_std = float(np.std(y_train_raw, ddof=0))
133 if not y_std > 0:
134     raise ValueError(...)
135 y_train_scaled = (y_train_raw - y_mean) / y_std
136 y_validation_scaled = (y_validation_raw - y_mean) / y_std
137 y_test_scaled = (y_test_raw - y_mean) / y_std
```

- **第 131 行**：只对训练目标求平均数，并用 `float` 转为普通小数。
- **第 132 行**：`np.std` 求标准差。`ddof=0` 表示按任务指定的“总体标准差”计算。标准差衡量目标值有多分散。
- **第 133–134 行**：读作“如果标准差不大于 0，就报错”。常量目标没有可用的标准差，拿它做除数会出错，所以先守住这个底线。
- **第 135 行**：标准化公式 `(原数 - 训练均值) / 训练标准差`。均值以上得到正数，均值以下得到负数。
- **第 136–137 行**：验证、测试仍使用训练集的 `y_mean`、`y_std`，绝不重新计算。

### 4.4 第 139–144 行：把数据转成 PyTorch 张量

```python
139 def as_float_tensor(values: np.ndarray, columns: int) -> torch.Tensor:
141     tensor = torch.tensor(values, dtype=torch.float32)
142     if columns == 1 and tensor.dim() == 1:
143         tensor = tensor.view(-1, 1)
144     return tensor
```

- **第 139 行**：这是 `prepare_diabetes` 内部的小助手。`values` 是要转换的数组；`columns` 告诉它期望有几列。
- **第 141 行**：把 NumPy 数组变成 `float32` 张量。神经网络权重默认也是 `float32`，两边类型一致才能计算。
- **第 142 行**：`tensor.dim()` 数维度。如果目标原本是一维 `(N,)`，且期望列数是 1，就进入下一行。
- **第 143 行**：`.view(-1,1)` 改形状。`-1` 表示“第一维由 PyTorch 自动推算”；1 表示固定成一列。所以 `(265,)` 变成 `(265,1)`。
- **第 144 行**：交回张量。

### 4.5 第 146–162 行：装进 `DiabetesData`

这一段把已准备好的每一个变量按名字塞入 `DiabetesData(...)`。每一行形如 `字段名=变量名`，左边是总文件夹的格子名，右边是刚才算出的内容。第 150–155 行特别调用 `as_float_tensor(...)`：输入保持 `(N,10)`，标签保证 `(N,1)`；第 161 行的 `x_frame.columns.tolist()` 把表头变成普通文字列表。

这里没有训练、没有 `backward()`、没有优化器。它只负责数据准备，这正是组件拆分的意义。

## 5. `make_train_loader`：把训练集切成小批（第 165–185 行）

```python
175 dataset = TensorDataset(data.x_train_tensor, data.y_train_scaled)
176 generator = torch.Generator()
177 generator.manual_seed(shuffle_seed)
178 return DataLoader(
179     dataset,
180     batch_size=batch_size,
181     shuffle=True,
182     drop_last=False,
183     num_workers=0,
184     generator=generator,
185 )
```

- **第 175 行**：`TensorDataset` 把“第 i 行特征”与“第 i 行标签”用同一个索引捆绑。以后打乱时它们会一起移动，不会错配。
- **第 176–177 行**：造一个独立的随机数发生器，再固定种子。SGD、Adam 两组分别新建并使用相同种子，因此每一轮看到的样本顺序相同，比较更公平。
- **第 178–185 行**：`DataLoader` 是发货员。`batch_size=32` 表示每次给训练函数 32 条；`shuffle=True` 表示每轮打乱；`drop_last=False` 表示最后不足 32 条的 9 条也不能丢；`num_workers=0` 表示由当前进程取数据，Windows 上最稳定；最后把刚固定的 `generator` 交给它。

## 6. `models.py`：MLP 长什么样（第 20–67 行）

```python
20  import torch
21  from torch import nn
24  N_FEATURES = 10
25  DEFAULT_HIDDEN_DIM = 16
49  def build_mlp(hidden_dim: int = DEFAULT_HIDDEN_DIM,
50                input_dim: int = N_FEATURES) -> nn.Sequential:
63      return nn.Sequential(
64          nn.Linear(input_dim, hidden_dim),
65          nn.ReLU(),
66          nn.Linear(hidden_dim, 1),
67      )
```

- **第 24 行**：本数据集每条样本固定有 10 个输入数。
- **第 25 行**：正式实验默认生成 16 个隐藏特征。这个数叫“隐藏层宽度”。
- **第 49–52 行**：`hidden_dim=DEFAULT_HIDDEN_DIM` 是默认参数；不特别指定时宽度为 16。`input_dim=N_FEATURES` 同理，默认输入为 10 维。
- **第 63 行**：`nn.Sequential` 把括号内的层按从上到下的顺序连成流水线。上层输出自动成为下层输入。
- **第 64 行**：`nn.Linear(input_dim, hidden_dim)`。若输入是一批 `(B,10)`，输出为 `(B,16)`。它对应公式 `z=W1x+b1`。`W1` 和 `b1` 都会学习。
- **第 65 行**：`nn.ReLU()` 对每个数单独做：负数变 0，正数原样保留。形状不变，仍是 `(B,16)`。没有这一层，两层 Linear 连起来仍只是一条大直线，表达不了非线性关系。
- **第 66 行**：`nn.Linear(hidden_dim,1)` 把 16 个隐藏特征合成一个预测，形状从 `(B,16)` 到 `(B,1)`。最后不加 ReLU，因为标准化后的目标可以是负数；若加 ReLU，所有偏低的预测都会被硬切成 0。
- **第 67 行**：关闭 `Sequential` 的括号，交回完整模型。

### 宽度从 16 改为 8，为什么必须改两处？

当 `hidden_dim=8` 时，第 64 行自动变成 `Linear(10,8)`，第 66 行同时自动变成 `Linear(8,1)`。这就是把 `hidden_dim` 同时传给两处的好处。

若只手工把第一层改成 `Linear(10,8)`，第二层仍写 `Linear(16,1)`，第一层交出 8 个数，第二层却坚持要 16 个数，像“8 根插头要插进 16 孔插座”，必然报尺寸不匹配错误。

## 7. `forward_shape_trace`：把每层形状打印出来（第 70–86 行）

```python
80  shapes: list[tuple[int, ...]] = [tuple(x.shape)]
81  current = x
82  with torch.no_grad():
83      for layer in model:
84          current = layer(current)
85          shapes.append(tuple(current.shape))
86  return shapes
```

- **第 80 行**：先建立一个列表，把原输入形状放进去。`tuple(x.shape)` 会把 `torch.Size([32,10])` 变成普通的 `(32,10)`，方便显示。
- **第 81 行**：`current` 是“当前正在流水线里流动的数据”，一开始等于输入 `x`。
- **第 82 行**：只查形状，不训练，因此关闭求导记录。
- **第 83 行**：`for layer in model` 按顺序取出 Sequential 中的第一层 Linear、ReLU、第二层 Linear。
- **第 84 行**：当前数据经过当前层，并把新结果重新放回 `current`。
- **第 85 行**：每经过一层，就把新形状追加到列表。
- **第 86 行**：交回完整形状链。例如 `(32,10) → (32,16) → (32,16) → (32,1)`。

## 8. `parameter_count`：数一数模型有多少可学习旋钮（第 89–101 行）

这段依次查看模型中的每层；`isinstance(layer, nn.Linear)` 只对 Linear 层计数，ReLU 没有可学习参数所以跳过。`numel()` 是“有多少个数字”。

- 宽度 16：第一层权重 `10×16=160`、偏置 `16`；第二层权重 `16×1=16`、偏置 `1`；总数 `193`。
- 宽度 8：第一层 `10×8+8=88`；第二层 `8×1+1=9`；总数 `97`。

参数更多不等于必然更好：容量更大，也更容易在小数据上记住训练集而不适应新数据。所以本周只做宽度检查，不偷偷增加参数搜索。

## 本部分的口头检查

你应能不看代码回答：

1. `(32,10)` 中 32 和 10 分别是什么意思？
2. 为什么 `StandardScaler` 只能在训练集上 `fit`？
3. 为什么目标标签要从 `(N,)` 变为 `(N,1)`？
4. 为什么两层 Linear 中间必须有 ReLU？
5. 为什么最后输出层不加 ReLU？
6. 宽度 16 改成 8 时，哪两层需要同步改、为什么？

---

# 第三部分：小批量训练、Adam、评价与最佳轮次

这一部分回答最容易混淆的几个问题：

- 为什么最终输出只有一个损失，第一层也能收到梯度？
- 为什么一轮训练结束不能直接拿最后一批的 `loss` 当成绩？
- SGD 和 Adam 具体分别在哪里配置？
- 为什么训练 200 轮，最后却可能用第 88 轮的模型？

主文件是 `training.py` 和 `evaluation.py`；优化器设置在 `optimizers.py`。

## 1. `optimizers.py`：第 22–37 行，两套实验配置

```python
22  MLP_OPTIMIZER_CONFIGS = (
23      {"name": "sgd", "label": "MLP + SGD",
24       "learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0},
30      {"name": "adam", "label": "MLP + Adam",
31       "learning_rate": 0.001, "momentum": 0.0, "weight_decay": 0.0},
37  )
```

- **第 22 行**：圆括号包住两份字典，形成一个元组。元组可看成“不准备随手改的固定清单”。
- **第 23、30 行的 `name`**：程序内部用的短名字，后面用于判断该造 SGD 还是 Adam。
- **`label`**：给人看的名称，画图、输出时显示 `MLP + SGD`、`MLP + Adam`。
- **第 24 行**：SGD 的学习率为 `0.01`。
- **第 31 行**：Adam 的学习率为 `0.001`。二者不同，因此这次实验严格说是在比较“两套完整配置”，不应把差异全归咎于 Adam 或 SGD 本身。
- **`momentum=0.0`**：任务要求的 SGD 不使用动量。字典里也为 Adam 保留这个字段，只是 `build_optimizer` 不把它传给 Adam。
- **`weight_decay=0.0`**：不做权重衰减。这样本周把注意力放在基础训练流程上。

### SGD 和 Adam 到底差在哪里？

若某个参数当前值为 `w`，反向传播给它的当前梯度为 `g`：

- SGD 的直觉是 `w ← w - 学习率 × g`：本次看见向东的建议，就向东走一步。
- Adam 会保存历史梯度的“平均方向”和“波动大小”，对每个参数分别调节步幅：稳定、方向一致的建议更可信；忽大忽小的方向会更谨慎。

Adam 不是“不需要学习率”。学习率仍决定总体走路速度；Adam 主要决定不同参数之间怎样分配步子。

## 2. `training.py` 第 125–142 行：一个批次的五步训练

```python
125 def train_one_batch(model, xb, yb, loss_fn, optimizer) -> float:
137     optimizer.zero_grad()
138     prediction = model(xb)
139     loss = loss_fn(prediction, yb)
140     loss.backward()
141     optimizer.step()
142     return float(loss.item())
```

- **第 125–131 行**：函数收到五样东西。`xb` 中的 `b` 是 batch，表示“当前小批输入”；`yb` 是同一批答案。这里不是整份 265 条训练集，而通常是 32 条。
- **第 137 行**：清掉上一批留下的 `.grad`。PyTorch 默认累积梯度；若遗漏，第二批梯度会变成“第一批加第二批”，不再是我们想要的当前批次梯度。
- **第 138 行**：把当前批特征穿过 `Linear → ReLU → Linear`。若 `xb` 是 `(32,10)`，`prediction` 是 `(32,1)`。
- **第 139 行**：将当前批预测与当前批标签比较，算一个标量 MSE。当前批是哪 32 条，这一次梯度就只由哪 32 条决定。
- **第 140 行**：`backward()` 从这个最终损失开始反向传播。PyTorch 在第 138 行前向时已经悄悄记住各层的计算关系；反传时按链式法则一路回去，给最后 Linear、ReLU 前的第一 Linear 都写好梯度。
- **第 141 行**：优化器读取每个参数的 `.grad` 并更新参数。用 SGD 就按当前梯度走；用 Adam 就使用当前与历史梯度状态。
- **第 142 行**：将这个批次的损失转成普通小数交回。它是在第 141 行更新**前**计算出的，因此不能当更新后的整轮训练 MSE。

### “第一层为什么也能学？”的白话答案

最后损失只直接看到预测值，但预测值来自第二层；第二层输入来自 ReLU；ReLU 输入来自第一层。像水从第一层流到最后，误差的影响也能沿着管道倒着追溯。`loss.backward()` 自动做的就是这份“倒查责任”：第一层权重改变一点，会让最终损失变多少？这个“会变多少”就是第一层的梯度。

## 3. `train_one_epoch`：第 145–166 行，一轮等于遍历所有批次

```python
145 def train_one_epoch(model, loader, loss_fn, optimizer) -> dict[str, float]:
158     model.train()
159     batch_losses: list[float] = []
160     for xb, yb in loader:
161         batch_losses.append(train_one_batch(model, xb, yb, loss_fn, optimizer))
162     return {
163         "mean_batch_loss": float(np.mean(batch_losses)),
164         "last_batch_loss": float(batch_losses[-1]),
165         "batch_count": float(len(batch_losses)),
166     }
```

- **第 145 行**：收到 `loader`，而不是直接收到完整训练张量。`loader` 每次会交出一对 `(xb,yb)`。
- **第 158 行**：`model.train()` 切回训练模式。当前网络只有 Linear 与 ReLU，训练/评价时数值相同；但以后如果加入 Dropout、BatchNorm，这一行就决定它们是否按训练规则工作，因此必须保留。
- **第 159 行**：建立空列表，准备记下每一批的损失。
- **第 160 行**：`for xb, yb in loader` 的意思是“每从 DataLoader 拿一批，就同时取特征和它配对的标签”。训练集共有 265 条，所以 32 条一批时会循环 9 次。
- **第 161 行**：每批调用前节五步函数，并把它返回的批次损失追加到列表。这个动作会在每一批都真正修改模型参数。
- **第 162–166 行**：交回三个统计量。`np.mean(batch_losses)` 是所有批次损失的简单平均；`batch_losses[-1]` 的 `-1` 在 Python 中表示列表最后一个元素；`len` 是批次数。

**为什么最后一批损失不能当本轮训练 MSE？**

1. 它只看了最后 9 条样本，不能代表全部 265 条。
2. 它在最后一批执行 `optimizer.step()` 前算出；整轮结束时模型的参数已经又更新了一次。
3. 各批大小不同，简单平均的批次损失也不等于严格按每条样本平均的完整训练 MSE。

所以每一轮之后必须另行用整份训练集重新预测、重新算 MSE。

## 4. `BestModel`：第 169–175 行，保存“最佳时刻”的快照

```python
169 @dataclass
170 class BestModel:
173     epoch: int | None = None
174     validation_mse: float = float("inf")
175     state_dict: dict[str, torch.Tensor] | None = None
```

- `epoch` 记录最佳轮号，一开始不知道，所以是 `None`。
- `float("inf")` 是正无穷大。任何正常有限的验证 MSE 都小于它，因此第一轮一定能成为第一个“当前最佳”。
- `state_dict` 是模型权重与偏置的完整字典快照。若只记轮号、不保存这份快照，训练继续后参数会变，最终无法回到当时最好的模型。

## 5. `fit_mlp`：第 178–230 行，完整训练与选最佳轮次

```python
199 history: list[dict[str, object]] = []
200 best = BestModel()
202 for epoch in range(1, epochs + 1):
203     batch_stats = train_one_epoch(model, loader, loss_fn, optimizer)
206     train_mse = evaluation.mse_scaled(model, data.x_train_tensor, data.y_train_scaled)
207     validation_mse = evaluation.mse_scaled(
208         model, data.x_validation_tensor, data.y_validation_scaled
209     )
211     history.append({...})
223     if validation_mse < best.validation_mse:
224         best.epoch = epoch
225         best.validation_mse = validation_mse
226         best.state_dict = copy.deepcopy(model.state_dict())
228 if best.epoch is not None:
229     history[best.epoch - 1]["is_best"] = True
230 return history, best
```

- **第 199 行**：建立空的历史记录列表，最终包含每一轮的一行数据。
- **第 200 行**：新建“目前最佳模型”的账本；初始验证误差是无穷大。
- **第 202 行**：`range(1, epochs+1)` 从 1 循环到 200。Python 的 `range` 不包含结尾，所以要写 `epochs+1`。
- **第 203 行**：跑一轮小批量训练；这一步会九次左右改模型。
- **第 206 行**：这一轮全结束后，用**现在这一刻的模型**预测完整训练集，计算标准化目标尺度上的训练 MSE。
- **第 207–209 行**：用同一个模型、完全不更新参数的方式评价完整验证集。验证集的答案用于挑模型，而不参与 `backward()`。
- **第 211–220 行**：加一条字典记录。`config`、`learning_rate` 说明是哪组实验；`epoch` 是轮号；`train_MSE`、`validation_MSE` 是整份数据重新评价的结果；`last_batch_loss` 特意保留供你对照；`target_scale` 防止别人把标准化 MSE 当原始单位；`is_best` 先填 `False`。
- **第 223 行**：只有当新验证 MSE **严格更小**才替换。这让完全相同的 MSE 保留更早的轮次，符合任务要求。
- **第 224–225 行**：更新最佳轮号与最佳验证 MSE。
- **第 226 行**：`model.state_dict()` 取当前所有权重和偏置；`copy.deepcopy` 深复制一份。深复制像拍照片：以后模型继续训练、权重继续变化，照片不会跟着变。
- **第 228–229 行**：训练完后，回到那条最佳轮次记录，把 `is_best` 改成 `True`，画图时可找到它。
- **第 230 行**：同时交回 200 条历史与最佳快照。

## 6. `load_best_state`：第 233–241 行，回到最佳轮次

```python
239 if best.state_dict is None:
240     raise RuntimeError("没有记录到最佳轮次参数")
241 model.load_state_dict(best.state_dict)
```

- **第 239–240 行**：安全检查。如果没有保存过最佳参数，不允许假装恢复成功。
- **第 241 行**：把模型当前最后一轮的参数，替换为那张最佳快照里的参数。注意这不是继续训练，不会改优化器状态；只是让模型回到验证集最好的那一时刻。

## 7. 基线：第 247–264 行，必须有“不用神经网络”的对照

```python
247 def fit_mean_baseline(y_train_raw) -> float:
253     return float(np.mean(y_train_raw))
256 def fit_linear_baseline(x_train_scaled, y_train_raw) -> LinearRegression:
262     model = LinearRegression()
263     model.fit(x_train_scaled, y_train_raw)
264     return model
```

- 均值基线永远预测训练集目标平均值，不看任何特征。若复杂模型都打不过它，说明训练很可能出了问题。
- 线性回归基线使用已经标准化的输入和原始单位目标。它不需要 PyTorch，却是判断 MLP 是否真的有价值的重要参照。

## 8. `evaluation.py`：评价时只预测，不学习

### 第 21–34 行：纯预测 `predict_tensor`

```python
21  def predict_tensor(model, x) -> np.ndarray:
31      model.eval()
32      with torch.no_grad():
33          output = model(x)
34      return output.detach().cpu().numpy()
```

- **第 31 行**：`model.eval()` 切换评价模式。现在模型没有 Dropout 或 BatchNorm，所以结果看不出差别，但这是以后防错的规范。
- **第 32 行**：不需要梯度，关闭自动求导。这样少占内存，也确保评价不会偷偷留下梯度。
- **第 33 行**：只是前向预测；没有 `loss.backward()`，更没有 `optimizer.step()`，所以参数不会改变。
- **第 34 行**：`detach()` 脱离计算图；`cpu()` 保证张量在 CPU；`numpy()` 转为 NumPy，方便后面算指标。

### 第 37–45 行：标准化尺度上的 MSE

```python
43 prediction = predict_tensor(model, x)
44 target = y_scaled.detach().cpu().numpy()
45 return float(np.mean((prediction - target) ** 2))
```

- 预测与标准化目标相减得到误差；`** 2` 是平方；`np.mean` 是平均；所以正是 MSE。
- 这用于画训练/验证曲线和选最佳轮次，不能与原始单位 RMSE 直接比数字。

### 第 48–54 行：还原原始尺度

```python
54 return prediction_scaled * y_std + y_mean
```

训练时是 `(原值-均值)/标准差`，反过来就是“乘回标准差，再加回均值”。这一步让 `-1.2` 这类标准化预测重新变成人能理解的疾病指标预测。

### 第 57–75 行：MAE、MSE、RMSE

```python
65 y_true = np.asarray(y_true, dtype=float).reshape(-1)
66 y_prediction = np.asarray(y_prediction, dtype=float).reshape(-1)
67 if y_true.shape != y_prediction.shape:
68     raise ValueError(...)
71 error = y_prediction - y_true
72 mae = float(np.mean(np.abs(error)))
73 mse = float(np.mean(error ** 2))
74 rmse = float(np.sqrt(mse))
75 return {"MAE": mae, "MSE": mse, "RMSE": rmse}
```

- **第 65–66 行**：先确保两边都是一维小数数组。`reshape(-1)` 表示不管原来是 `(N,)` 还是 `(N,1)`，都排成一行 N 个数。
- **第 67–70 行**：形状不同就报错，防止拿 88 个预测去和 89 个真实值比较。
- **第 71 行**：预测减真实值，得到每条样本的误差。
- **第 72 行**：`np.abs` 取绝对值，平均后是 MAE；单位与原目标一样。
- **第 73 行**：平方再平均是 MSE；大错误会被处罚更重，单位变成“原目标的平方”。
- **第 74 行**：`sqrt` 开根号得到 RMSE，单位回到原目标，便于比较。
- **第 75 行**：用字典把三项指标贴上名字交回。

### 第 78–101 行：MLP 的原始单位评价

`evaluate_mlp_raw` 的第 89–91 行严格依次做三件事：取标准化预测、用训练集的 `y_mean/y_std` 还原、同原始标签算指标。`evaluate_mlp_scaled` 的第 94–100 行只是一层薄包装，复用前面的标准化 MSE 函数来核对最佳轮次曲线。

## 本部分的口头检查

1. 当前 `xb, yb` 是整份训练集还是当前批？为什么下一批梯度可能不同？
2. `backward()` 与 `step()` 各自改变了什么？
3. 为什么最后一批损失不等于整轮训练 MSE？
4. `model.eval()` 与 `torch.no_grad()` 分别解决什么问题？
5. Adam 为什么仍然需要学习率？
6. 为什么要深复制 `state_dict`，而不是只记最佳轮号？

---

# 第四部分：入口、结果文件、画图与组会讲解

`run_mlp.py` 是实践二的总指挥。它自己不重新实现数据切分、模型、训练或画图，而是按顺序调用组件。这正是本周“按职责拆文件”的要求。

## 1. `run_mlp.py` 顶部：导入与固定设置（第 16–36 行）

```python
16  from __future__ import annotations
18  import copy
19  import sys
21  import numpy as np
22  import pandas as pd
23  import torch
24  from torch import nn
26  import data
27  import evaluation
28  import models
29  import optimizers
30  import reporting
31  import training
34  EPOCHS = 200
35  LOSS_FUNCTION = nn.MSELoss()
```

- `copy` 用于复制初始网络，确保 SGD 与 Adam 从一模一样的参数起跑。
- `sys` 仅用于 Windows 终端 UTF-8 中文输出。
- `np` 用于空测试指标写 `NaN`、生成均值基线的固定预测；`pd` 用于把记录变成表。
- `torch` 用于固定随机种子；`nn` 用于建立 MSE 损失。
- 第 26–31 行分别导入六个职责单一的组件。入口只“安排工作”，核心实现不堆在这里。
- `EPOCHS = 200` 表示每组都训练满 200 轮；`LOSS_FUNCTION = nn.MSELoss()` 是整周回归任务的损失规则。

## 2. 两个小显示函数（第 39–62 行）

`use_utf8_output()` 与实践一相同，作用是避免中文乱码。`describe_shapes` 与 `print_forward_check` 只把形状排成可读文字，例如：

```text
输入(32, 10) → Linear(32, 16) → ReLU(32, 16) → Linear(32, 1)
```

第 50 行：`["输入"] + [type(layer).__name__ for layer in model]` 是“列表推导式”。它先放一个 `输入`，再依次取模型中的每层，拿其类型名称。`zip(names, shapes)` 把名称与同位置形状配成对；`" → ".join(...)` 再用箭头连接。它们只帮助讲解，不影响训练。

## 3. `build_metric_row`（第 65–92 行）：统一结果表的一行

```python
65  def build_metric_row(*, model_name, display_name, learning_rate, best_epoch,
66                       train_metrics, validation_metrics, test_metrics,
67                       selected="", note="") -> dict[str, object]:
78      return {
79          "model": model_name,
83          "train_MAE": train_metrics["MAE"],
84          "train_RMSE": train_metrics["RMSE"],
85          "validation_MAE": validation_metrics["MAE"],
86          "validation_RMSE": validation_metrics["RMSE"],
87          "test_MAE": np.nan if test_metrics is None else test_metrics["MAE"],
88          "test_RMSE": np.nan if test_metrics is None else test_metrics["RMSE"],
89          "selected_mlp": selected,
90          "target_scale": "original",
91          "note": note,
92      }
```

- 第 65 行开头的单独 `*` 表示后面的参数调用时必须写名字，例如 `model_name="adam_mlp"`。这样十多个参数不容易放错位置。
- 第 78–92 行建立一个字典，即 CSV 表格的一行。
- 第 83–86 行从评价函数给回的指标字典中，按名称取 MAE、RMSE。
- 第 87–88 行是条件表达式：若 `test_metrics is None`，就写 `np.nan`（表格中的空值）；否则取测试指标。任务规定只有选中的一组 MLP 能看测试集，所以未选中 MLP 必须留空。
- 第 90 行明确所有 MAE、RMSE 都已在原始目标单位上算好，避免把训练曲线的标准化 MSE 混进来。

## 4. `main()` 第 95–119 行：准备数据并检查前向形状

```python
95  def main() -> None:
96      use_utf8_output()
97      reporting.ensure_results_dir()
103     dataset = data.prepare_diabetes()
115     torch.manual_seed(data.SEED)
116     probe_model = models.build_mlp(hidden_dim=models.DEFAULT_HIDDEN_DIM)
117     demo_batch = dataset.x_train_tensor[:32]
118     print_forward_check(probe_model, demo_batch, ...)
```

- 第 96–97 行先处理输出编码并确保 `results/` 目录存在。
- 第 103 行一次调用数据组件，得到封装了三份数据、标准化统计量和特征名的 `dataset`。
- 第 115 行在建模型前固定 PyTorch 随机种子。网络权重最初是随机的；固定种子才能复现。
- 第 116 行建立一个宽度 16 的临时模型，仅供形状检查。
- 第 117 行 `[:32]` 取训练张量前 32 行，得到一批演示数据。冒号表示“从开头取到第 32 行之前”。
- 第 118 行显示每层形状。没有 `backward()` 和 `step()`，因此不训练。

## 5. 第 120–165 行：让 SGD、Adam 在公平条件下训练

```python
123 torch.manual_seed(data.SEED)
124 initial_model = models.build_mlp(...)
126 histories = []
127 best_records = {}
128 trained_models = {}
131 for config in optimizers.MLP_OPTIMIZER_CONFIGS:
136     model = copy.deepcopy(initial_model)
137     optimizer = optimizers.build_optimizer(...)
144     loader = data.make_train_loader(..., shuffle_seed=data.SEED)
148     history, best = training.fit_mlp(...)
158     histories.extend(history)
159     best_records[name] = best
160     trained_models[name] = model
```

- 第 123–124 行只建一次“原始未训练网络”。
- 第 126–128 行建三个空容器：`histories` 收集 400 条曲线记录；`best_records` 以 `sgd`、`adam` 为钥匙保存各自最佳快照；`trained_models` 保存两台训练过的模型。
- 第 131 行逐一遍历优化器配置。一次处理 SGD，一次处理 Adam。
- 第 132–134 行从配置字典取出内部名字、人类标签、学习率，并转为确定类型。
- 第 136 行深复制同一个初始网络。若直接重复使用同一个 `model`，Adam 会从 SGD 已经训练过的终点起跑，比较就不公平。
- 第 137–142 行按当前名称建出对应优化器。
- 第 144 行为每组新建 DataLoader，并给相同打乱种子。两组第 1 轮、2 轮……拿到的样本顺序一致。
- 第 148–157 行正式训练 200 轮，得到该配置的 200 条历史与最佳快照。
- 第 158 行的 `extend` 与 `append` 不同：`append` 会把 200 条历史当成一个大盒子塞进去；`extend` 会把这 200 条逐条添入总列表，最终正好 400 行。
- 第 159–160 行分别按名字保存最佳记录和训练结束后的模型对象。

## 6. 第 168–177 行：专门证明“最后一批 loss 不等于整轮 MSE”

```python
169 last_epoch_rows = history_frame[history_frame["epoch"] == EPOCHS]
171 for row in last_epoch_rows.itertuples(index=False):
172     print(... row.last_batch_loss ... row.train_MSE ...)
```

- 第 166 行先用 `pd.DataFrame(histories)` 将字典列表变为二维表。
- 第 169 行筛选 `epoch == 200` 的两行，即 SGD、Adam 各自最后一轮。
- 第 171 行 `itertuples` 一行一行读表；`index=False` 不把 pandas 的行号塞进来。
- 第 172–176 行把末批损失与整份训练 MSE 打印出来，并写明两个差异来源：样本范围不同、参数时刻不同。

## 7. 第 179–234 行：恢复最佳参数、做基线、严格使用测试集

```python
181 for name, best in best_records.items():
183     training.load_best_state(model, best)
185     "train": evaluation.evaluate_mlp_raw(...)
188     "validation": evaluation.evaluate_mlp_raw(...)
195 baseline_value = training.fit_mean_baseline(...)
196 linear_model = training.fit_linear_baseline(...)
217 selected_name = min(("sgd", "adam"), key=lambda candidate: (...))
223 test_metrics = {
224     "mean_baseline": ...,
227     "linear_regression": ...,
230     f"{selected_name}_mlp": ...,
234 }
```

- 第 181 行 `.items()` 依次交出字典里的“名字、最佳记录”两个东西。
- 第 182–183 行取回相应模型并恢复最佳轮次参数。之后算的训练、验证 RMSE 都是该最佳轮，而不是最后 200 轮。
- 第 185–192 行调用原始尺度评价：模型预测标准化目标后，内部会还原单位，再和原始标签比。
- 第 195 行拟合均值基线；第 196 行拟合线性回归基线。
- 第 198–213 行分别在训练、验证集上算两种基线指标。`np.full(样本数, baseline_value)` 表示“造一个与样本数一样长、每个格子都填均值”的预测数组。
- 第 217–220 行 `min(...)` 在 `sgd`、`adam` 中选择验证 RMSE 更小的一组。`lambda candidate:` 是一个短小的“给每个候选者打分”的函数；评分元组的第二项 `candidate != "sgd"` 实现同分时优先 SGD。
- 第 223–234 行只为均值基线、线性回归、选中的 MLP 计算测试指标。另一台 MLP 不读取测试集，这是防止用测试集反复挑模型。

## 8. 第 236–273 行：组装四行 `metrics.csv`

`metric_rows` 是一个列表，里面固定四个字典：均值基线、线性回归、SGD MLP、Adam MLP。

- 第 238–247 行调用 `build_metric_row` 填均值基线；`best_epoch`、`learning_rate` 留空，因为它不训练神经网络。
- 第 248–257 行填线性回归，同样没有“最佳轮次”。
- 第 259–273 行循环填两个 MLP。第 267 行放各自最佳轮；第 270 行对未选 MLP 从测试字典拿不到值，因此自动留空；第 271 行给选中的那一行写 `yes`。
- 第 274 行将四行转成 DataFrame，留给保存和画图。

## 9. 第 276–286 行：隐藏层宽度从 16 改为 8 的检查

```python
278 narrow_model = models.build_mlp(hidden_dim=8)
279 narrow_shapes = models.forward_shape_trace(narrow_model, demo_batch)
281 wide_shapes = models.forward_shape_trace(initial_model, demo_batch)
283 print(models.parameter_count(narrow_model))
284 print(models.parameter_count(initial_model))
```

- 第 278 行新建宽度 8 网络。它与正式训练模型无关。
- 第 279 行仅做前向形状检查，得到 `(32,10)→(32,8)→(32,8)→(32,1)`。
- 第 281 行同时显示宽度 16 形状，方便对照。
- 第 283–284 行打印参数量 97 与 193。没有训练，没有新图表，符合“只做检查”的要求。

## 10. `reporting.py`：只写文件、只画图（第 12–132 行）

### 第 12–26 行：固定结果目录

```python
12 from pathlib import Path
14 import matplotlib
16 matplotlib.use("Agg")
17 import matplotlib.pyplot as plt
18 import pandas as pd
20 RESULTS_DIR = Path(__file__).parent / "results"
23 def ensure_results_dir() -> Path:
25     RESULTS_DIR.mkdir(parents=True, exist_ok=True)
26     return RESULTS_DIR
```

- `Path` 用来安全拼接路径；`__file__` 是当前 `reporting.py` 的位置；`.parent` 是它所在的 `week05` 文件夹；`/ "results"` 是在该处加上子目录。
- `matplotlib.use("Agg")` 选择只生成 PNG 的后端，不要求电脑有弹窗图形界面。
- 第 25 行的 `parents=True` 表示上级目录缺失时可一起建立；`exist_ok=True` 表示已存在也不报错。

### 第 32–50 行：三个 CSV 保存函数

```python
34 path = ensure_results_dir() / filename
35 pd.DataFrame(rows).round(6).to_csv(path, index=False, encoding="utf-8-sig")
36 return path
```

三个函数结构相同，只是默认文件名不同。

- `rows` 是已算好的字典列表；`pd.DataFrame(rows)` 变为表。
- `.round(6)` 统一保留六位小数，避免 CSV 出现难读的浮点尾巴。
- `.to_csv` 写文件；`index=False` 不加无意义的 0、1、2 行号；`utf-8-sig` 让 Windows Excel 打开中文列名通常不会乱码。
- 交回 `path`，入口就能打印“文件保存在哪里”。这里没有模型、更没有训练循环。

### 第 56–100 行：`loss_curves.png`

这段先按配置得到两个子图：左 SGD、右 Adam。

- 第 68 行 `dict.fromkeys(...)` 利用字典不重复的性质，按出现顺序取两个配置名。
- 第 69 行 `plt.subplots(1, len(configs), ...)` 建 1 行、2 列图；`sharex=True` 使两图横轴轮数一致。
- 第 73–92 行循环处理每个配置：第 74 行筛选并按轮号排序；第 75–78 行分别画训练 MSE、验证 MSE；第 80–86 行用绿色虚线和圆点标出该配置的最佳验证轮；第 88–92 行添加标题、坐标轴、网格和图例。
- 第 94–99 行设总标题、整理空白、以 160 dpi 保存图片、`plt.close` 关闭图对象避免连续运行时积累内存。

### 第 103–126 行：`rmse_comparison.png`

- 第 109 行取四个显示名；第 110 行做 0、1、2、3 四个柱子位置；第 111 行设单组柱宽。
- 第 114–117 行分别画训练与验证的两组柱，并通过 `value - width/2`、`value + width/2` 让它们在同一个模型标签两侧并排。
- 第 118–125 行设置横轴文字、纵轴原始单位、标题、网格、图例、保存并关闭。

## 11. 第 289–324 行：保存、汇总和入口开关

```python
290 history_path = reporting.save_history(histories)
291 metrics_path = reporting.save_metrics(metric_rows)
295 curves_path = reporting.plot_loss_curves(history_frame, best_epochs)
299 rmse_path = reporting.plot_rmse_comparison(plot_frame)
305 summary = metrics_frame[[...]]
309 reporting.print_table(..., summary)
318 for path in (history_path, metrics_path, curves_path, rmse_path):
319     print(f"  {path}")
324 if __name__ == "__main__":
325     main()
```

- 第 290–291 行保存两张表；第 295、299 行保存两张图。它们接收的是前面已经算好的结果，不重新训练。
- 第 293–294 行用字典推导式建立“显示标签 → 最佳轮号”的对照表，供损失曲线标线。
- 第 297–299 行只挑出柱状图需要的三列，`.copy()` 复制成可单独使用的小表。
- 第 305–309 行挑出控制台最需要看的列，调用 `print_table` 只显示它们。
- 第 318–319 行循环打印四个生成文件的完整路径。
- 第 324–325 行仍是入口保护：直接执行本文件才训练；导入它不会自动开始。

## 12. 结果文件应如何读

| 文件 | 看什么 | 不该误解成什么 |
|---|---|---|
| `step_trace.csv` | 三阶段权重、偏置、梯度、更新前后损失 | 反传会改权重；实际改参数的是 `step()` |
| `history.csv` | 两组各 200 轮的整份训练/验证标准化 MSE | 原始单位的 RMSE |
| `metrics.csv` | 四组模型的原始单位 MAE/RMSE、最佳轮、测试列 | 两台 MLP 都可以看测试集 |
| `loss_curves.png` | 是否收敛、最佳轮在哪里、训练/验证差距 | 数值可同原始单位 RMSE 直接比较 |
| `rmse_comparison.png` | 四个模型在训练/验证集上的原始单位 RMSE | MLP 必然比线性回归强 |

## 13. 组会可直接照着说的版本

“我把任务拆成六个组件和两个入口。`data.py` 只负责 60/20/20 划分、训练集标准化和 DataLoader；`models.py` 定义 `10→16→1` 的 MLP；`optimizers.py` 集中放 SGD 和 Adam 配置。训练的核心在 `training.py`：每个 batch 都是先 `zero_grad`，再前向、算 MSE、`backward` 求所有层的梯度、`step` 更新参数。每轮结束后我在 `evaluation.py` 用 `eval` 和 `no_grad` 对整份训练集、验证集重新评价，并保存验证 MSE 最低时的 `state_dict`。最后恢复最佳轮次参数，将预测还原到原始单位后算 MAE/RMSE。`reporting.py` 只负责保存三张 CSV 和两张 PNG，不再训练模型。”

若被追问 Adam：

“SGD 主要依据当前梯度走一步；Adam 同时利用历史梯度方向与波动来调节不同参数的步幅，但仍要学习率控制总体步长。两组学习率不同，因此实验比较的是两套具体配置。”

---

## 附录：`requirements.txt` 的每一行

这个文件不是训练代码，而是环境清单。每行的 `==` 在这里**不是** Python 的比较符，而是“安装时必须使用这个准确版本”的意思。

- `numpy==2.5.3`：数字数组与均值、标准差等基础计算工具。
- `pandas==3.0.6`：表格数据工具；本项目主要把结果整理成表格时使用。
- `scikit-learn==1.9.1`：提供 diabetes 数据集、数据划分、标准化器和线性回归基线。
- `matplotlib==3.11.2`：负责把训练历史和 RMSE 画成 PNG 图。
- `scipy==1.18.1`：scikit-learn 需要的底层科学计算依赖。
- `torch==2.14.1`：本作业的核心深度学习工具箱，负责张量、模型、自动求导和优化器。

至此，项目中所有可执行 Python 文件与环境清单，都已经在这一份讲义中按任务目标、名词、调用顺序和源代码行号解释完毕。

---

## 附录：`requirements.txt` 的每一行

这个文件不是训练代码，而是环境清单。每行的 `==` 在这里**不是** Python 的比较符，而是“安装时必须使用这个准确版本”的意思。

- `numpy==2.5.3`：数字数组与均值、标准差等基础计算工具。
- `pandas==3.0.6`：表格数据工具；本项目主要把结果整理成表格时使用。
- `scikit-learn==1.9.1`：提供 diabetes 数据集、数据划分、标准化器和线性回归基线。
- `matplotlib==3.11.2`：负责把训练历史和 RMSE 画成 PNG 图。
- `scipy==1.18.1`：scikit-learn 需要的底层科学计算依赖。
- `torch==2.14.1`：本作业的核心深度学习工具箱，负责张量、模型、自动求导和优化器。

至此，项目中所有可执行 Python 文件与环境清单，都已经在这一份讲义中按任务目标、名词、调用顺序和源代码行号解释完毕。
