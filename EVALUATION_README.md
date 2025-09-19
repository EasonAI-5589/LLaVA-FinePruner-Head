# 聚合策略综合评估系统

本评估系统用于系统性地测试和比较不同的注意力头聚合策略在LLaVA模型上的效果。

## 系统架构

### 1. 核心修改文件

#### 模型文件
- `llava/model/language_model/modeling_llama_fastv_fine_head.py`: 实现了5种不同的聚合策略
- `llava/model/language_model/llava_llama.py`: 添加了新的模型加载方式

#### 评估文件
- `llava/eval/model_vqa_loader.py`: 添加了聚合策略参数支持
- `scripts/v1_5/7b/mme.sh`: 示例脚本，支持策略参数传递

### 2. 聚合策略

| 策略名称 | 描述 | 特点 |
|---------|------|------|
| `original` | 原始方案 | 简单求和+平均 |
| `quality_based` | 基于注意力质量 | 峰值 × 差异性 |
| `entropy_based` | 基于注意力熵 | 选择最集中的注意力 |
| `topk_focus` | 基于Top-K关注度 | 关注最重要的visual token |
| `kurtosis_based` | 基于峰度 | 选择分布最尖锐的头 |

## 使用方法

### 快速测试

运行快速评估脚本，测试关键配置：

```bash
# 设置GPU并运行快速评估
export CUDA_VISIBLE_DEVICES=0,1,2,3
./eval_strategies_quick.sh
```

这将测试：
- 3种策略：original, quality_based, entropy_based
- 3个任务：mme, vqav2, gqa
- 3种配置：192_16, 128_16, 64_8

### 完整评估

运行完整的评估脚本，测试所有组合：

```bash
# 设置GPU并运行完整评估
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./eval_comprehensive.sh
```

这将测试：
- 5种策略：所有策略
- 10个任务：所有主要任务
- 多种配置：不同token和head数量组合

### 单独测试特定策略

你也可以直接使用修改后的脚本测试特定配置：

```bash
# 测试quality_based策略
bash scripts/v1_5/7b/mme.sh fastv+finepruner+head 192 16 quality_based

# 测试entropy_based策略
bash scripts/v1_5/7b/vqav2.sh fastv+finepruner+head 128 8 entropy_based
```

## 结果分析

### 自动分析

运行结果分析脚本：

```bash
# 分析评估结果
python analyze_results.py --result_dir ./evaluation_results/YYYYMMDD_HHMMSS
```

这将生成：
- 详细的JSON报告 (`analysis_report.json`)
- 文本摘要报告 (`summary_report.txt`)
- 可视化图表 (在 `figures/` 目录下)

### 手动分析

结果文件结构：
```
playground/data/eval/TASK/answers/llava_TASK/CKPT/METHOD/vtn_TOKEN_HEAD_STRATEGY/
├── merge.jsonl          # 合并的评估结果
├── 8_0.jsonl           # 分块结果文件
├── 8_1.jsonl
└── ...
```

## 配置参数说明

### 命令行参数

- `--pruning_method`: 使用 `fastv+finepruner+head` 来启用头选择功能
- `--visual_token_num`: visual token数量 (64, 128, 192)
- `--H`: 注意力头数量 (8, 16, 24, 32)
- `--aggregation_strategy`: 聚合策略名称

### 环境变量

- `CUDA_VISIBLE_DEVICES`: 指定使用的GPU
- `CKPT_DIR`: 模型检查点目录
- `DATA_DIR`: 评估数据目录

## 预期结果

根据初步分析，不同策略的预期表现：

1. **quality_based**: 在大多数任务上应该比original有改进
2. **entropy_based**: 在需要精确注意力的任务上表现更好
3. **topk_focus**: 在visual reasoning任务上可能有优势
4. **kurtosis_based**: 最精确但计算开销稍大

## 故障排除

### 常见问题

1. **ImportError**: 确保已安装所有依赖
```bash
pip install transformers torch torchvision matplotlib seaborn pandas numpy
```

2. **CUDA内存不足**: 减少GPU数量或batch size
```bash
export CUDA_VISIBLE_DEVICES=0,1  # 使用更少GPU
```

3. **结果文件不存在**: 检查路径配置和权限
```bash
# 检查数据目录
ls -la /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/
```

### 调试模式

启用调试输出：
```bash
# 添加调试标志
python -m llava.eval.model_vqa_loader --debug ...
```

## 扩展功能

### 添加新策略

1. 在 `modeling_llama_fastv_fine_head.py` 中添加新的 `elif` 分支
2. 在参数选择中添加新策略名称
3. 更新评估脚本

### 添加新任务

1. 在评估脚本中添加新的任务配置
2. 确保数据文件路径正确
3. 添加任务特定的后处理逻辑

## 性能优化建议

1. **并行评估**: 使用多GPU并行加速
2. **批量处理**: 合理设置batch size
3. **结果缓存**: 避免重复计算已有结果
4. **增量评估**: 只评估新增的配置组合

## 联系方式

如有问题或建议，请联系开发团队或提交Issue。
