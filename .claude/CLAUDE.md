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

## 核心技术：智能求同存异的动态Head选择策略

### 设计理念

基于"求同存异"思想的全面升级版本，解决传统方法的5大核心问题：
1. **智能求同**：策略质量加权的共识识别，而非简单投票
2. **精准存异**：多层次差异化选择，确保功能空间全覆盖
3. **自适应阈值**：根据策略一致性动态调整共识标准
4. **复杂度感知**：基于attention pattern复杂度智能确定头数量
5. **质量驱动**：全流程基于策略分辨能力和attention质量优化

这种方法真正实现了**数据驱动、自适应、智能化**的visual head筛选。

### 核心算法：Intelligent Consensus-Diversity Selection

#### Step 1: 策略质量自适应评估

对每个策略在当前attention pattern下进行质量评估，确定其可信度：

```python
for strategy in all_strategies:
    scores = compute_strategy_scores(image_attention, strategy)

    # 多维质量评估
    discriminability = scores.std()  # 分数区分度
    balance = 1.0 - (score_range / score_mean).clamp(0, 1)  # 分布合理性
    correlation = F.cosine_similarity(scores, attention_intensity).abs()  # 与真实强度相关性

    quality = 0.4 * discriminability + 0.3 * balance + 0.3 * correlation

    # 根据质量确定精细化选择数量
    if quality > 0.7:
        selection_count = int(H * 0.3)  # 高质量策略精选模式
    elif quality > 0.4:
        selection_count = H // 2        # 标准模式
    else:
        selection_count = int(H * 0.65) # 广撒网模式
```

#### Step 2: 动态共识阈值与质量加权投票

根据策略间一致性动态调整共识标准，用策略质量进行加权投票：

```python
# 计算策略间一致性
strategy_consistency = compute_overlap_ratio_across_strategies()

# 自适应阈值调整
if consistency > 0.6:
    dynamic_threshold = min(0.5, base_threshold * (1 + consistency))  # 高一致性提高阈值
elif consistency < 0.2:
    dynamic_threshold = max(0.15, base_threshold * consistency * 2)   # 低一致性降低阈值
else:
    dynamic_threshold = base_threshold  # 标准阈值

# 质量加权投票
for strategy, result in strategy_selections.items():
    weight = result['quality']  # 策略质量作为投票权重
    for head_idx in result['indices']:
        head_weighted_votes[head_idx] += weight
```

#### Step 3: 复杂度感知的头数量智能确定

基于attention pattern复杂度，智能调整所需头数量：

```python
# 多维复杂度评估
head_diversity = 1 - avg_similarity_between_heads     # 头间多样性
attention_entropy = normalized_entropy_across_heads   # 分布复杂度
attention_variance = normalized_variance_of_values    # 数值方差

complexity = 0.4 * head_diversity + 0.3 * attention_entropy + 0.3 * attention_variance

# 复杂度自适应调整
if complexity > 0.7:
    target_count = int(gap_based * (1.2 + 0.3 * complexity))  # 高复杂度需要更多头
elif complexity < 0.3:
    target_count = int(gap_based * (0.7 + 0.6 * complexity))  # 低复杂度需要较少头
else:
    target_count = gap_based  # 中等复杂度
```

#### Step 4: 多层次差异化头选择

确保功能空间的全面覆盖，避免冗余：

```python
# 层次1: 与共识头的差异性
diversity_from_consensus = 1 - cosine_similarity(available_heads, consensus_heads)

# 层次2: 候选头之间的多样性
for each_remaining_slot:
    if first_diversity_head:
        select_max_diversity_from_consensus()
    else:
        # 综合考虑与共识头和已选头的差异性
        diversity_from_selected = 1 - cosine_similarity(candidate, selected_diversity_heads)
        total_score = 0.6 * diversity_from_consensus + 0.4 * diversity_from_selected
        select_max_total_score()
```

#### Step 5: 智能权重聚合

基于头类型和质量的精细化权重分配：

```python
for head_idx in final_indices:
    if head_idx in consensus_heads:
        # 共识头权重：基于选择它的策略质量均值
        avg_quality = mean([strategy_quality for strategy that selected this head])
        weight = 1.0 + 0.5 * avg_quality
    else:
        # 差异化头权重：基于其attention质量
        attention_quality = normalized_attention_sum(head_idx)
        weight = 0.8 + 0.4 * attention_quality

weights = softmax(all_weights)
```

### 核心优势

#### 1. 策略质量自适应评估
- **创新点**：不再平等对待所有策略，而是评估每个策略在当前pattern下的有效性
- **技术细节**：基于分数区分度、分布合理性、与真实attention的相关性进行多维评估
- **实际效果**：高质量策略获得更多投票权重和更精准的选择数量

#### 2. 动态共识阈值机制
- **突破点**：摒弃固定1/3阈值，根据策略间一致性自动调整共识标准
- **自适应逻辑**：高一致性场景提高阈值（精选模式），低一致性场景降低阈值（包容模式）
- **智能化水平**：真正实现数据驱动的阈值优化

#### 3. 复杂度感知的头数量调整
- **核心理念**：attention pattern复杂度决定所需头数量，而非盲目固定值
- **评估维度**：头间多样性 + 分布熵 + 数值方差的综合复杂度指标
- **动态范围**：1-32全范围支持，复杂pattern使用更多头，简单pattern使用较少头

#### 4. 多层次差异化选择
- **设计精髓**：不仅考虑与共识头的差异，还确保差异化头之间的多样性
- **算法层次**：层次1解决consensus-diversity差异，层次2解决diversity内部多样性
- **功能覆盖**：确保选中的头覆盖不同的visual功能空间，避免冗余

#### 5. 智能权重聚合机制
- **权重逻辑**：共识头基于策略质量获得权重奖励，差异化头基于attention质量获得权重
- **平衡设计**：既体现共识头的广泛认可，又保证差异化头的独特贡献
- **精细化水平**：每个头的权重都是基于其具体贡献动态计算

### 实验预期效果

基于智能求同存异的5大技术突破，预期获得显著性能提升：

#### 性能提升预测
- **策略质量自适应**：相比平等投票提升1.5-2.5%
- **动态共识阈值**：相比固定阈值提升0.8-1.2%
- **复杂度感知调整**：相比固定头数提升1.0-1.5%
- **多层次差异化**：相比简单差异化提升0.5-1.0%
- **智能权重聚合**：相比固定权重提升0.3-0.8%

**综合预期提升：3-6%**，在复杂多模态理解任务上效果更显著

#### 稳定性改进
- **鲁棒性**：策略质量评估确保在不同attention pattern下的稳定表现
- **自适应性**：动态阈值和复杂度感知避免over-fitting到特定场景
- **可解释性**：共识机制提供明确的头选择依据和质量评估

#### 效率优化
- **计算效率**：复杂度感知避免不必要的头数量浪费
- **存储效率**：精细化选择减少冗余头的存储开销
- **推理效率**：智能权重聚合提高attention融合质量

### 使用方法

```bash
# 启用智能求同存异的动态头选择
python -m llava.eval.model_vqa_loader \
    --pruning_method ablation_a \
    --enable-dynamic-selection \
    --visual_token_num 128 \
    --H 16  # 参考值，实际会基于复杂度在1-32范围内智能调整
```

### 核心参数说明

- `--enable-dynamic-selection`: 启用智能求同存异的5维优化策略
- `--visual_token_num`: 视觉token数量 (64/128/192/576)
- `--H`: 头数量参考值，算法会基于attention复杂度智能确定最终数量

### 实验验证

#### 基础验证配置
```bash
# 验证1: 标准配置 (预期智能选择10-20个heads)
--visual_token_num 128 --enable-dynamic-selection

# 验证2: 高复杂度场景 (预期智能选择16-28个heads)
--visual_token_num 192 --enable-dynamic-selection

# 验证3: 低复杂度场景 (预期智能选择6-14个heads)
--visual_token_num 64 --enable-dynamic-selection
```

#### 高级调试模式

启用详细调试信息查看智能选择过程：
```python
# 在config中添加
model.config.debug_mode = True
```

#### 调试输出示例
```
🎯 Intelligent Consensus-Diversity: 17 heads
   Consensus: 11 | Diversity: 6
   Dynamic threshold: 0.28 | Complexity: 0.73
   Strategy qualities: sparsity=0.85, hierarchical=0.72, graph_based=0.45...
   Selected consensus heads: [2, 7, 12, 16, 19, 23, 28, 30, 31, 14, 25]
   Selected diversity heads: [5, 9, 18, 22, 26, 29]
```

### 性能监控

添加性能指标监控：
```python
# 可选：添加attention质量监控
model.config.monitor_attention_quality = True
```

这将输出：
- 策略质量分布
- 共识头vs差异化头的贡献比例
- attention复杂度变化趋势
- 头数量自适应轨迹
