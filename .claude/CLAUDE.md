# LLaVA-FinePruner-Head 项目说明

这是一个基于LLaVA的视觉多模态大语言模型优化项目，主要研究视觉token pruning和attention head选择等策略。希望通过有效的注意力头筛选策略，准确的关注到visual head，从而进一步有效的指导剪枝工作，在剪枝的同时实现模型推理加速的效果

## 项目概述

本项目实现并对比了了多种Visual Token Pruning的方法：

- **FastV**: 基于attention score的token选择
- **SparseVLM**: 基于text2visual 注意力分数进行token剪枝
- **PDrop**: 基于Decoding Layer层数进行的剪枝
- VisPruner(待完成，完成过后更新文档)
- CDPruner(待完成，完成过后更新文档)
- **FastV+FinePruner**: FastV结合细粒度head选择

## 主要脚本

- `eval.sh`: 完整的模型评估脚本，包含FastV/SparseVLM/PDrop对比模型和对比方法测试
- `eval_head.sh`: 头选择策略消融研究评估脚本，是我们实验的主要
- `enhanced_head_selection.py`: 增强的头选择策略实现

### 核心代码

- `llava/model/language_model/modeling_llama_fastv_fine_head.py`: 核心模型实现，包含动态头选择
- `llava/eval/`: 评估相关代码
- `scripts/v1_5/7b/`: 各数据集评估脚本

### 实验结果

- `head_strategy_ablation/`: 头选择策略消融研究结果
- `visualize_head_ablation.py`: 结果可视化脚本

## 常用命令

### 运行完整评估

```bash
bash eval.sh  # 运行FastV/SparseVLM/PDrop完整评估
```

### 运行特定方法评估

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/7b/[dataset].sh [method] [tokens] [heads]
```

### 头选择策略消融研究

```bash
bash eval_head.sh
```

## 评估数据集

支持以下10个数据集的评估：

- VQAv2, GQA, VizWiz, SQA, TextVQA
- POPE, MME, MMBench, MMBench-CN, MMVet

## 参数配置

- **tokens**: 视觉token数量 (64, 128, 192, 576)
- **heads**: attention head数量 (8, 16, 24, 32)
- **方法**: vanilla, fastv, sparsevlm, pdrop, fastv+finepruner

## 环境要求

- CUDA 8卡GPU环境
- PyTorch + transformers
- LLaVA dependencies

## 核心技术：基于预实验数据的优化动态Head选择策略

### 设计理念

基于大量预实验数据分析（35个配置×7种策略×4个数据集），我们发现不同token数量下最优策略存在明显模式。设计了一个数据驱动的自适应选择策略，能够根据资源约束自动选择最优策略组合并动态确定头数量。

### 预实验关键发现

根据head_strategy_ablation实验分析报告：

**策略优势统计：**
- **sparsity**: 在13个配置中表现最佳（高token数优势明显）
- **graph_based**: 在10个配置中表现最佳（低token数表现突出）
- **hierarchical**: 在推理任务(GQA/POPE)中表现优异
- **top_k_sum**: 在8个head配置下经常最优（极度压缩场景）

**最优配置示例：**
- TextVQA: Token=192, Head=16, Strategy=sparsity → 58.02%
- MME: Token=192, Head=16, Strategy=sparsity → 1853.7
- GQA: Token=192, Head=16, Strategy=hierarchical → 57.78%
- POPE: Token=192, Head=16, Strategy=hierarchical → 77.15%

### 优化策略框架

#### Step 1: 资源感知策略筛选

基于token数量（R值）确定候选策略：

```python
# 高资源场景 (R≥166, 对应192 tokens)
if self.R >= 166:
    candidate_strategies = ['sparsity', 'hierarchical']
    preferred_head_range = (14, 26)  # 支持非固定值

# 中等资源场景 (R≥98, 对应128 tokens)
elif self.R >= 98:
    candidate_strategies = ['sparsity', 'hierarchical', 'top_k_sum', 'max_attention']
    preferred_head_range = (8, 26)

# 低资源场景 (R<98, 对应64 tokens及以下)
else:
    candidate_strategies = ['graph_based', 'multi_objective', 'attention_range']
    preferred_head_range = (6, 18)
```

#### Step 2: 多策略并行评估

对每个候选策略计算质量分数并确定最优头数量：

```python
# 并行评估所有候选策略
for strategy in candidate_strategies:
    scores = compute_strategy_scores(image_attention, strategy)
    optimal_count = determine_optimal_head_count(scores, preferred_head_range)
    quality_score = evaluate_combination_quality(selected_attention, scores)

    # 记录策略结果
    strategy_results[strategy] = {
        'indices': selected_indices,
        'count': optimal_count,
        'quality': quality_score
    }
```

#### Step 3: 动态头数量优化

支持10、12、18、20等非传统固定值：

```python
# 方法1: 拐点检测 - 寻找质量显著下降点
gaps = sorted_scores[:-1] - sorted_scores[1:]
significant_gaps = gaps > (mean_gap + 0.8 * std_gap)
gap_based_count = first_significant_gap_position

# 方法2: 质量阈值 - 筛选高质量heads
quality_threshold = mean_score - 0.3 * std_score
quality_based_count = (scores > threshold).sum()

# 方法3: 累积贡献 - 达到85%贡献度
cumulative_contribution = torch.cumsum(normalized_scores, dim=0)
contribution_based_count = (cumulative <= 0.85).sum() + 1

# 综合决策：取三种方法的中位数
optimal_count = median([gap_based, quality_based, contribution_based])
```

#### Step 4: 组合质量综合评估

多维度评估头选择组合：

```python
# 质量指标1: 多样性 (heads间差异性)
diversity = 1 - avg_similarity_between_heads

# 质量指标2: 分数分布合理性
score_quality = (score_std + score_range) / 2

# 质量指标3: 注意力分布有效性
attention_quality = attention_variance * attention_max

# 综合质量评分
overall_quality = 0.3*diversity + 0.3*score_quality + 0.4*attention_quality
```

### 核心优势

1. **数据驱动**：基于35×7×4=980个实验配置的结果指导策略选择
2. **资源自适应**：根据token数量自动选择最优策略组合
3. **动态头数量**：支持6-26范围内的精细调整，不限于8/16/24
4. **多策略融合**：整合sparsity的效率、hierarchical的推理能力、graph_based的低资源优势

### 实验预期效果

基于预实验数据预测：
- 高资源场景：sparsity策略预期性能提升2-3%
- 中等资源场景：多策略融合预期稳定性提升
- 低资源场景：graph_based策略预期在64 tokens下优化显著
- 动态头数量：相比固定值预期额外获得0.5-1%性能提升

### 使用方法

```bash
# 启用优化的动态头选择
python -m llava.eval.model_vqa_loader \
    --pruning_method ablation_a \
    --enable-dynamic-selection \
    --debug-mode True \
    --visual_token_num 128 \
    --H 16  # 作为参考值，实际会动态调整
```

### 实验验证

推荐在以下配置下验证效果：
```bash
# 高性能验证 (应选择sparsity + 16-18 heads)
--visual_token_num 192

# 平衡验证 (应选择sparsity/hierarchical + 12-20 heads)
--visual_token_num 128

# 效率验证 (应选择graph_based + 8-14 heads)
--visual_token_num 64
```
