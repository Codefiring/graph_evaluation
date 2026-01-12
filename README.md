# 图评估算法

用于评估驱动ioctl状态转换图的预测结果。

## 功能

该程序实现了基于序列的图评估算法，通过比较ground truth和预测结果的状态转换图，计算序列级的precision和recall。

支持两种使用模式：
- **单文件评估**：评估单个驱动的预测结果
- **批量评估**：通过配置文件批量评估多个驱动

## 算法说明

1. **序列生成**：从初始状态出发，遍历所有长度≤k的ioctl调用序列
2. **序列比较**：计算两个图的序列集合的交集
3. **评估指标**：
   - **Precision**: 预测结果中合法的序列，有多少在ground truth中也允许
   - **Recall**: ground truth允许的序列，有多少被预测结果也允许
   - **F1 Score**: Precision和Recall的调和平均

## 使用方法

### 方式一：批量评估（推荐）

使用配置文件批量评估多个驱动：

```bash
python batch_evaluator.py config.json
```

#### 配置文件格式

创建JSON格式的配置文件（参考 `examples/config_example.json`）：

```json
{
  "parameters": {
    "max_length": 5,
    "use_sampling": false,
    "sample_size": 10000,
    "max_sequences": null
  },
  "drivers": [
    {
      "name": "driver1",
      "gt_file": "data/driver1_gt.txt",
      "pred_file": "data/driver1_pred.txt"
    },
    {
      "name": "driver2",
      "gt_file": "data/driver2_gt.txt",
      "pred_file": "data/driver2_pred.txt",
      "max_length": 6,
      "use_sampling": true,
      "sample_size": 50000
    }
  ]
}
```

**配置说明**：
- `parameters`: 全局参数，所有驱动共享（可被驱动特定参数覆盖）
  - `max_length`: 最大序列长度k
  - `use_sampling`: 是否使用采样方法
  - `sample_size`: 采样数量
  - `max_sequences`: 最大序列数量限制
- `drivers`: 驱动列表
  - `name`: 驱动名称（用于结果文件命名）
  - `gt_file`: Ground truth文件路径
  - `pred_file`: 预测结果文件路径
  - 每个驱动可以覆盖全局参数（可选）

#### 批量评估参数

```bash
python batch_evaluator.py <config_file> [选项]

选项:
  --output-dir, -o    输出目录 (默认: results)
  --format, -f        输出格式: json, csv, 或 both (默认: both)
  --quiet, -q         静默模式，不输出详细信息
```

#### 输出结果

批量评估会在输出目录生成以下文件：
- `{driver_name}_result.json`: 每个驱动的详细评估结果（JSON格式）
- `all_results_{timestamp}.json`: 所有驱动的完整结果汇总（JSON格式）
- `summary_{timestamp}.csv`: 所有驱动的摘要结果（CSV格式，便于Excel分析）

### 方式二：单文件评估

评估单个驱动的预测结果：

```bash
python evaluate_single.py <gt_file> <pred_file> [选项]
```

#### 参数说明

- `gt_file`: Ground truth文件路径
- `pred_file`: 预测结果文件路径
- `--max-length, -k`: 最大序列长度k（默认: 5）
- `--sampling`: 使用采样方法（当分支太多时）
- `--sample-size`: 采样数量（默认: 10000）
- `--max-sequences`: 最大序列数量限制（用于控制内存）

#### 示例

```bash
# 基本评估，最大序列长度为5
python evaluate_single.py gt.txt pred.txt --max-length 5

# 使用采样方法，适合分支很多的情况
python evaluate_single.py gt.txt pred.txt --max-length 6 --sampling --sample-size 50000

# 限制最大序列数量，避免内存溢出
python evaluate_single.py gt.txt pred.txt --max-length 5 --max-sequences 100000
```

## 输入文件格式

每行格式为：`"old_state","ioctl","new_state"` (CSV格式，字段用双引号包围)

示例：
```
"NPU_VERTEX_OPEN","npu_vertex_s_graph","NPU_VERTEX_GRAPH"
"NPU_VERTEX_GRAPH","npu_vertex_streamon","NPU_VERTEX_STREAMON"
"NPU_VERTEX_STREAMOFF","npu_vertex_close","NPU_VERTEX_CLOSED"
```

注意：
- 字段用逗号分隔
- 每个字段必须用双引号包围（标准CSV格式）
- 程序使用Python的csv模块解析，能正确处理引号内的特殊字符
- 空行会被自动跳过
- 字段前后的空白会被自动去除

## 输出

### 单文件评估输出

程序会输出以下评估指标：
- **序列级 Precision**: 预测结果中合法序列在ground truth中的比例
- **序列级 Recall**: ground truth中序列被预测结果覆盖的比例
- **F1 Score**: Precision和Recall的调和平均
- **详细统计**: 包括序列数量、假阳性、假阴性等

### 批量评估输出

#### JSON格式结果

每个驱动的详细结果包含：
```json
{
  "driver_name": "driver1",
  "precision": 0.5417,
  "recall": 1.0000,
  "f1_score": 0.7027,
  "gt_sequences": 13,
  "pred_sequences": 24,
  "common_sequences": 13,
  "false_positives": 11,
  "false_negatives": 0,
  "gt_stats": {
    "states": 5,
    "ioctls": 4,
    "transitions": 6
  },
  "pred_stats": {
    "states": 5,
    "ioctls": 4,
    "transitions": 7
  },
  "gt_file": "data/driver1_gt.txt",
  "pred_file": "data/driver1_pred.txt"
}
```

#### CSV格式结果

摘要CSV文件包含以下列：
- `driver_name`: 驱动名称
- `precision`: Precision值
- `recall`: Recall值
- `f1_score`: F1分数
- `gt_sequences`: Ground truth序列数
- `pred_sequences`: 预测序列数
- `common_sequences`: 共同序列数

## 项目结构

```
graph_evaluation/
├── graph_evaluator.py      # 核心评估模块
├── batch_evaluator.py      # 批量评估程序
├── evaluate_single.py      # 单文件评估程序
├── README.md               # 本文件
├── requirements.txt        # 依赖（仅使用标准库）
├── .gitignore              # Git忽略文件
├── data/                   # 数据文件目录
│   ├── example_gt.txt      # 示例ground truth文件
│   └── example_pred.txt    # 示例预测文件
├── examples/               # 示例配置目录
│   └── config_example.json # 配置文件示例
└── results/                # 结果输出目录（自动创建，已加入.gitignore）
```

## 注意事项

1. 当状态转换图的分支很多时，建议使用`--sampling`选项进行采样，避免内存和时间消耗过大
2. 初始状态会自动检测（选择没有入边的状态，或第一个出现的状态）
3. 如果两个图都没有状态转换，程序会返回precision和recall为0
4. 批量评估时，如果某个驱动的文件不存在，会跳过该驱动并继续处理其他驱动
5. CSV文件使用UTF-8编码，可以在Excel中正常打开（可能需要指定编码）

## 快速开始

1. 准备配置文件（参考 `examples/config_example.json`）
2. 运行批量评估：
   ```bash
   python batch_evaluator.py config.json
   ```
3. 查看结果：
   - 详细结果：`results/{driver_name}_result.json`
   - 汇总结果：`results/all_results_{timestamp}.json`
   - CSV摘要：`results/summary_{timestamp}.csv`
