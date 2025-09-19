# LLaVA-FinePruner-Head 消融研究实验配置

## 实验设计概述

我们设计了两个独立的消融研究来分别验证：
1. **消融研究A**: 注意力头筛选方法的效果
2. **消融研究B**: Query Token选择策略的效果

## 消融研究A: 注意力头筛选方法

### 实验目标
固定使用最后一个query token，比较不同注意力头筛选方法对visual token重要性评估的影响。

### 实验配置
**文件**: `modeling_llama_fastv_ablation_a.py`

**固定设置**:
- Query token: 最后一个token (`last_attention[0, :, -1, :]`)
- 其他参数: K=2, R根据模型规模确定

**变量**: `head_selection_strategy`

#### 实验组
1. **`sum`** (Baseline)
   - 方法: 简单求和 `head_attention = image_attention.sum(dim=-1)`
   - 说明: 原始FastV方法

2. **`variance`**
   - 方法: 基于注意力方差 `head_variance = image_attention.var(dim=-1)`
   - 假设: 方差大的头对visual token有更集中的关注

3. **`entropy`**
   - 方法: 基于注意力熵，选择熵值较小的头
   - 假设: 低熵头有更集中、更有信息量的注意力分布

4. **`max_attention`**
   - 方法: 基于最大注意力值
   - 假设: 有强烈关注点的头更重要

5. **`attention_range`**
   - 方法: 基于注意力分布范围 (max - min)
   - 假设: 范围大的头有更好的区分能力

6. **`sparsity`**
   - 方法: 基于L2范数衡量稀疏性
   - 假设: 稀疏注意力更有针对性

7. **`top_k_sum`**
   - 方法: 基于top-k注意力值总和
   - 假设: 重点关注最重要的几个visual token

8. **`weighted_quality`**
   - 方法: 综合质量评分 (集中度 + 尖锐度 + 覆盖范围)
   - 假设: 多指标综合评估更准确

9. **`gini_coefficient`**
   - 方法: 基于基尼系数衡量不均匀性
   - 假设: 注意力分布不均匀的头更有区分性

### 实验命令示例
```bash
# 运行不同头筛选策略
python eval_model.py --head_selection_strategy sum --config ablation_a_config.json
python eval_model.py --head_selection_strategy variance --config ablation_a_config.json
python eval_model.py --head_selection_strategy entropy --config ablation_a_config.json
# ... 其他策略
```

---

## 消融研究B: Query Token选择策略

### 实验目标
固定使用某种头筛选方法，比较不同query token选择策略对visual token重要性评估的影响。

### 实验配置
**文件**: `modeling_llama_fastv_ablation_b.py`

**固定设置**:
- 头筛选方法: `variance` (基于消融研究A的结果选择)
- 其他参数: K=2, R根据模型规模确定

**变量**: `query_selection_strategy`

#### 实验组
1. **`last_token`** (Baseline)
   - 方法: 使用最后一个token
   - 说明: 原始FastV方法

2. **`last_n_tokens`** (n=3)
   - 方法: 使用最后3个token的平均注意力
   - 假设: 多个近期token提供更稳定的信号

3. **`last_5_tokens`**
   - 方法: 使用最后5个token的平均注意力
   - 假设: 更大的窗口可能捕获更多上下文

4. **`text_tokens_only`**
   - 方法: 仅使用text token对visual token的注意力
   - 假设: Text token更能反映语义相关性

5. **`weighted_recent`**
   - 方法: 对最近的token使用递增权重
   - 假设: 近期token更重要，但历史token也有价值

6. **`system_and_text`**
   - 方法: 使用system prompt + text token (排除visual token)
   - 假设: 所有非visual token都有助于理解

7. **`attention_weighted`**
   - 方法: 基于注意力强度对query token加权
   - 假设: 注意力强的token更重要

8. **`max_attention_token`**
   - 方法: 选择对visual token注意力最强的单个token
   - 假设: 最相关的单个token最重要

9. **`adaptive_by_length`**
   - 方法: 根据序列长度自适应选择策略
   - 假设: 不同长度需要不同策略

### 实验命令示例
```bash
# 运行不同query选择策略
python eval_model.py --query_selection_strategy last_token --fixed_head_selection variance --config ablation_b_config.json
python eval_model.py --query_selection_strategy text_tokens_only --fixed_head_selection variance --config ablation_b_config.json
python eval_model.py --query_selection_strategy attention_weighted --fixed_head_selection variance --config ablation_b_config.json
# ... 其他策略
```

---

## 实验评估指标

### 主要指标
1. **任务性能**
   - VQA准确率
   - MMBench分数
   - 其他多模态理解任务指标

2. **效率指标**
   - 推理速度 (tokens/sec)
   - 内存使用量
   - FLOPs计算量

3. **Visual Token质量**
   - 保留的visual token与ground truth的重叠度
   - 注意力分布的集中度
   - Visual token的多样性

### 实验条件
- **数据集**: 使用相同的评估数据集
- **模型**: LLaVA-7B/13B
- **硬件**: 统一的GPU环境
- **随机种子**: 固定种子确保可重现性

## 预期结果分析

### 消融研究A预期
- `variance`和`weighted_quality`可能表现较好
- `sum`作为baseline对比
- `entropy`可能在某些任务上有优势

### 消融研究B预期
- `text_tokens_only`和`attention_weighted`可能表现较好
- `adaptive_by_length`可能在不同长度文本上表现稳定
- 单token方法可能在简单任务上足够

## 实验执行计划

1. **阶段1**: 运行消融研究A，确定最佳头筛选方法
2. **阶段2**: 使用最佳头筛选方法运行消融研究B
3. **阶段3**: 结合最佳配置进行完整评估
4. **阶段4**: 错误分析和case study

这个设计可以清晰地分离两个改进维度的贡献，为后续的方法组合提供科学依据。