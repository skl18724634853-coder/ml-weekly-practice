# ETTh1 数据说明

本目录的 `ETTh1.csv` 来自 [ETDataset 官方仓库](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)。

第六周实际使用：

- 实践一：前 12 条小时记录的 `OT`；
- 实践二：前 14,400 条小时记录；
- 输入与目标：仅使用 `OT` 的历史值预测下一小时 `OT`；
- `HUFL`：只在验证首周图中作为事后观察的负载曲线，不输入预测模型。

原始列包含 `date`、六种负载通道与油温 `OT`。本周要求代码读取后检查：时间递增、无重复、相邻记录间隔一小时且所用数值无缺失。
