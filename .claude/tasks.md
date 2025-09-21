# 常见任务模板

## 1. 运行单个数据集评估

### 基础命令格式
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/[dataset].sh [method] [tokens] [heads]
```

### 示例命令
```bash
# VQAv2评估 - FastV方法，128个tokens
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/vqav2.sh fastv 128

# POPE评估 - FastV+FinePruner方法，192个tokens，16个heads
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/pope.sh fastv+finepruner 192 16

# MME评估 - SparseVLM方法，64个tokens
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/mme.sh sparsevlm 64
```

## 2. 批量评估任务

### 运行完整方法对比
```bash
# 运行所有方法的完整评估
bash eval.sh

# 运行头选择策略消融研究
bash eval_head.sh
```

### 自定义批量评估
```bash
# 评估特定方法在多个数据集上的表现
for task in pope mme vqav2 gqa; do
    for token in 192 128 64; do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/${task}.sh fastv $token
    done
done
```

## 3. 结果分析任务

### 解析评估结果
```bash
# 解析并汇总评估结果
python parse_evaluation_results.py

# 可视化头选择消融研究结果
python visualize_head_ablation.py
```

### 查看结果文件
```bash
# 查看POPE评估结果
cat head_strategy_ablation/head_strategy_pope_results.txt

# 查看MME评估结果
cat head_strategy_ablation/head_strategy_mme_results.txt
```

## 4. 代码开发任务

### 修改核心模型
```bash
# 编辑主要模型文件
vim llava/model/language_model/modeling_llama_fastv_fine_head.py

# 编辑头选择策略
vim enhanced_head_selection.py
```

### 代码检查和格式化
```bash
# Python类型检查
python -m mypy llava/ --ignore-missing-imports

# 代码格式化
black llava/ --line-length 100

# 代码风格检查
flake8 llava/ --max-line-length 100
```

## 5. 实验管理任务

### 创建新实验
```bash
# 创建新的消融研究目录
mkdir -p experiments/new_ablation/
mkdir -p experiments/new_ablation/results/

# 备份当前配置
cp eval_head.sh experiments/new_ablation/eval_config.sh
```

### 监控实验进度
```bash
# 监控GPU使用情况
nvidia-smi -l 1

# 查看实验日志
tail -f nohup.out

# 检查输出文件
ls -la playground/data/eval/*/answers/*/*/
```

## 6. 环境和依赖管理

### 检查环境
```bash
# 检查CUDA环境
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# 检查Python环境
which python
pip list | grep torch
```

### 数据准备
```bash
# 检查数据集路径
ls /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/

# 检查模型checkpoint
ls /mnt/bn/bes-mllm-shared/checkpoint/LLaVA/
```