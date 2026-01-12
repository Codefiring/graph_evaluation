# 图评估算法

用于评估驱动ioctl状态转换图的预测结果。

## 功能

该程序实现了基于序列的图评估算法，通过比较ground truth和预测结果的状态转换图，计算序列级的precision和recall。

## 算法说明

1. **序列生成**：从初始状态出发，遍历所有长度≤k的ioctl调用序列
2. **序列比较**：计算两个图的序列集合的交集
3. **评估指标**：
   - **Precision**: 预测结果中合法的序列，有多少在ground truth中也允许
   - **Recall**: ground truth允许的序列，有多少被预测结果也允许
   - **F1 Score**: Precision和Recall的调和平均

## 使用方法

### 基本用法

```bash
python graph_evaluator.py <gt_file> <pred_file> --max-length <k>
```

### 参数说明

- `gt_file`: Ground truth文件路径
- `pred_file`: 预测结果文件路径
- `--max-length, -k`: 最大序列长度k（默认: 5）
- `--sampling`: 使用采样方法（当分支太多时）
- `--sample-size`: 采样数量（默认: 10000）
- `--max-sequences`: 最大序列数量限制（用于控制内存）

### 示例

```bash
# 基本评估，最大序列长度为5
python graph_evaluator.py gt.txt pred.txt --max-length 5

# 使用采样方法，适合分支很多的情况
python graph_evaluator.py gt.txt pred.txt --max-length 6 --sampling --sample-size 50000

# 限制最大序列数量，避免内存溢出
python graph_evaluator.py gt.txt pred.txt --max-length 5 --max-sequences 100000
```

## 输入文件格式

每行格式为：`old_state,ioctl,new_state`

示例：
```
NPU_VERTEX_OPEN,"npu_vertex_s_graph","NPU_VERTEX_GRAPH"
NPU_VERTEX_FORMAT,"npu_vertex_streamoff","NPU_VERTEX_STREAMON"
```

注意：
- 字段用逗号分隔
- 字段可以用引号包围（引号会被自动去除）
- 空行会被忽略

## 输出

程序会输出以下评估指标：

- **序列级 Precision**: 预测结果中合法序列在ground truth中的比例
- **序列级 Recall**: ground truth中序列被预测结果覆盖的比例
- **F1 Score**: Precision和Recall的调和平均
- **详细统计**: 包括序列数量、假阳性、假阴性等

## 注意事项

1. 当状态转换图的分支很多时，建议使用`--sampling`选项进行采样，避免内存和时间消耗过大
2. 初始状态会自动检测（选择没有入边的状态，或第一个出现的状态）
3. 如果两个图都没有状态转换，程序会返回precision和recall为0
