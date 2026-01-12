# 项目结构说明

## 目录结构

```
graph_evaluation/
├── .gitignore              # Git忽略规则
├── README.md               # 项目说明文档
├── PROJECT_STRUCTURE.md    # 本文件：项目结构说明
├── requirements.txt        # Python依赖（仅使用标准库）
│
├── graph_evaluator.py      # 核心评估模块
│   ├── StateGraph          # 状态转换图类
│   ├── parse_graph_file()  # 解析状态转换文件
│   ├── evaluate_graphs()   # 执行评估计算
│   └── save_results_*()    # 结果保存功能
│
├── batch_evaluator.py      # 批量评估主程序
│   └── batch_evaluate()    # 批量评估函数
│
├── evaluate_single.py      # 单文件评估程序
│   └── 保持向后兼容的单文件评估接口
│
├── data/                   # 数据文件目录
│   ├── example_gt.txt       # 示例ground truth文件
│   └── example_pred.txt     # 示例预测文件
│   └── [你的驱动数据文件]   # 将实际的gt和pred文件放在这里
│
├── examples/               # 示例配置目录
│   └── config_example.json # 配置文件示例
│
└── results/                # 结果输出目录（自动创建，已加入.gitignore）
    ├── {driver_name}_result.json      # 每个驱动的详细结果
    ├── all_results_{timestamp}.json   # 所有驱动的完整汇总
    └── summary_{timestamp}.csv         # CSV格式摘要
```

## 文件说明

### 核心代码文件

- **graph_evaluator.py**: 包含所有核心评估逻辑，可被其他模块导入使用
- **batch_evaluator.py**: 批量评估入口，从配置文件读取任务并批量执行
- **evaluate_single.py**: 单文件评估入口，用于快速评估单个驱动

### 数据目录

- **data/**: 存放所有驱动的地面真值（ground truth）和预测结果文件
  - 建议按驱动名称组织子目录，如 `data/driver1/gt.txt`, `data/driver1/pred.txt`
  - 或直接使用文件名区分，如 `data/driver1_gt.txt`, `data/driver1_pred.txt`

### 示例目录

- **examples/**: 存放示例配置文件，供用户参考

### 结果目录

- **results/**: 评估结果输出目录
  - 由程序自动创建
  - 已加入 `.gitignore`，不会被版本控制跟踪

## 使用建议

1. **数据组织**：
   - 将所有驱动的数据文件放在 `data/` 目录下
   - 使用清晰的命名规则，如 `{driver_name}_gt.txt` 和 `{driver_name}_pred.txt`

2. **配置文件**：
   - 复制 `examples/config_example.json` 作为你的配置文件
   - 根据实际数据文件路径修改配置

3. **结果管理**：
   - 结果文件会自动保存在 `results/` 目录
   - 建议定期清理或归档旧的结果文件

4. **版本控制**：
   - 代码文件、配置示例、文档等会被跟踪
   - 数据文件和结果文件不会被跟踪（已在 `.gitignore` 中排除）

## Git 忽略规则

以下内容不会被版本控制跟踪：
- `__pycache__/` - Python缓存文件
- `*.pyc`, `*.pyo` - Python编译文件
- `results/` - 结果输出目录
- `*.csv`, `*.json` - 结果文件（除了示例配置）
- `test_*` - 测试文件
- IDE配置文件（`.vscode/`, `.idea/`等）
