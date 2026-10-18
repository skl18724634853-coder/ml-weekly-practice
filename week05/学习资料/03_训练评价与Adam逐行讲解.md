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
