import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, Cache, DynamicCache, \
    _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast


R_dict = {
    "7b": {
        2: {
            192: 166,
            128: 98,
            64: 30,
        },
        3: {
            192: 152,
            128: 82,
            64: 11,
        },
    },
    "13b": {
        2: {
            192: 172,
            128: 104,
            64: 37,
        },
        3: {
            192: 161,
            128: 92,
            64: 22,
        }
    }
}


class FineFastVLlamaModelAblationDynamic(LlamaModel):
    """
    简化版动态Head选择策略：基于实用性的策略选择
    去除复杂的多阶段处理，聚焦于有效的head选择方法
    """

    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super().__init__(config)
        self.system_prompt_length = 35
        self.visual_token_num = 0

        if config.num_hidden_layers == 32:
            self.scale = "7b"
        elif config.num_hidden_layers == 40:
            self.scale = "13b"

        if config.image_aspect_ratio == "pad":
            self.visual_token_length = 576
            self.anyres = False
        elif config.image_aspect_ratio == "anyres":
            self.visual_token_length = 2880
            self.anyres = True

        # FastV config
        self.K = 2
        self.R = R_dict[self.scale][self.K][visual_token_num]
        if self.anyres:
            self.R *= 5
        self.H = 32
        self.remove_sink = False

        # 简化的头选择策略配置
        self.head_selection_strategy = getattr(config, 'head_selection_strategy', 'sum')
        self.enable_dynamic_selection = getattr(config, 'enable_dynamic_selection', True)
        self.debug_mode = getattr(config, 'debug_mode', False)

        print(f"🔧 Dynamic Head Selection: strategy={self.head_selection_strategy}, enabled={self.enable_dynamic_selection}")

    def consensus_diversity_selection(self, image_attention):
        """
        智能求同存异策略：基于策略质量加权的共识-差异化选择

        核心创新：
        1. 策略质量自适应评估和加权投票
        2. 动态共识阈值根据策略一致性调整
        3. 精细化候选头选择基于策略分辨能力
        4. 多层次差异化确保功能空间覆盖
        5. attention复杂度感知的头数量调整
        """
        H, N = image_attention.shape

        # 所有候选策略
        all_strategies = ['sum', 'sparsity', 'hierarchical', 'graph_based',
                         'top_k_sum', 'max_attention', 'attention_range', 'multi_objective']

        # === 第一步：策略质量评估和自适应选择 ===
        strategy_selections = {}
        strategy_qualities = {}

        for strategy in all_strategies:
            scores = self.compute_strategy_scores(image_attention, strategy)

            # 评估策略在当前pattern下的质量
            quality = self.evaluate_strategy_quality(scores, image_attention)
            strategy_qualities[strategy] = quality

            # 根据策略分辨能力确定选择数量
            selection_count = self.adaptive_selection_count(scores, quality, H)
            selected_indices = scores.topk(k=selection_count).indices

            strategy_selections[strategy] = {
                'indices': selected_indices,
                'scores': scores[selected_indices],
                'quality': quality,
                'count': selection_count
            }

        # === 第二步：质量加权的共识头识别 ===
        head_weighted_votes = torch.zeros(H, device=image_attention.device)
        head_total_score = torch.zeros(H, device=image_attention.device)

        # 计算策略间的一致性
        strategy_consistency = self.compute_strategy_consistency(strategy_selections)

        # 动态调整共识阈值
        dynamic_threshold = self.adaptive_consensus_threshold(strategy_consistency, len(all_strategies))

        for strategy, result in strategy_selections.items():
            weight = result['quality']  # 策略质量作为投票权重
            for i, idx in enumerate(result['indices']):
                head_weighted_votes[idx] += weight
                head_total_score[idx] += result['scores'][i] * weight

        # === 第三步：复杂度感知的头数量确定 ===
        attention_complexity = self.compute_attention_complexity(image_attention)
        target_head_count = self.complexity_aware_head_count(
            image_attention, head_weighted_votes, attention_complexity
        )
        target_head_count = max(1, min(target_head_count, H))

        # === 第四步：共识头选择 ===
        consensus_heads = []
        consensus_scores = head_weighted_votes + 0.1 * head_total_score

        # 根据动态阈值选择共识头
        total_weight = sum(result['quality'] for result in strategy_selections.values())
        consensus_candidates = torch.where(head_weighted_votes >= dynamic_threshold * total_weight)[0]

        if len(consensus_candidates) > 0:
            consensus_sorted = consensus_candidates[torch.argsort(consensus_scores[consensus_candidates], descending=True)]
            consensus_heads = consensus_sorted[:min(target_head_count, len(consensus_sorted))].tolist()

        # === 第五步：多层次差异化头选择 ===
        remaining_count = target_head_count - len(consensus_heads)
        diversity_heads = []

        if remaining_count > 0:
            available_heads = [i for i in range(H) if i not in consensus_heads]
            if available_heads:
                diversity_heads = self.multi_level_diversity_selection(
                    image_attention, available_heads, consensus_heads, remaining_count
                )

        # === 第六步：智能权重聚合 ===
        final_indices = torch.tensor(consensus_heads + diversity_heads, device=image_attention.device)

        if len(final_indices) > 0:
            selected_attention = image_attention[final_indices]

            # 基于质量和类型的智能权重分配
            weights = self.compute_intelligent_weights(
                final_indices, consensus_heads, strategy_selections, image_attention
            )

            aggregated_attention = (selected_attention * weights.unsqueeze(-1)).sum(dim=0)
        else:
            # 保底策略
            fallback_idx = image_attention.sum(dim=-1).topk(k=1).indices
            final_indices = fallback_idx
            aggregated_attention = image_attention[fallback_idx[0]]

        if self.debug_mode:
            print(f"🎯 Intelligent Consensus-Diversity: {len(final_indices)} heads")
            print(f"   Consensus: {len(consensus_heads)} | Diversity: {len(diversity_heads)}")
            print(f"   Dynamic threshold: {dynamic_threshold:.3f} | Complexity: {attention_complexity:.3f}")

        return final_indices, aggregated_attention

    def evaluate_strategy_quality(self, scores, image_attention):
        """
        评估策略在当前attention pattern下的质量
        """
        # 质量指标1: 分数的区分度 (标准差)
        score_discriminability = scores.std()

        # 质量指标2: 分数分布的合理性 (避免极端值)
        score_range = scores.max() - scores.min()
        score_balance = 1.0 - (score_range / (scores.mean() + 1e-8)).clamp(0, 1)

        # 质量指标3: 与真实attention强度的相关性
        attention_intensity = image_attention.sum(dim=-1)  # 每个头的总attention
        correlation = F.cosine_similarity(scores, attention_intensity, dim=0).abs()

        # 综合质量分数
        quality = 0.4 * score_discriminability + 0.3 * score_balance + 0.3 * correlation
        return quality.item()

    def adaptive_selection_count(self, scores, quality, H):
        """
        根据策略分辨能力自适应确定选择数量
        """
        base_count = H // 2  # 基础选择数量

        # 高质量策略选择更少但更精准的头
        if quality > 0.7:
            return max(1, int(base_count * 0.6))  # 精选模式
        elif quality > 0.4:
            return base_count  # 标准模式
        else:
            return min(H, int(base_count * 1.3))  # 广撒网模式

    def compute_strategy_consistency(self, strategy_selections):
        """
        计算策略间的一致性
        """
        all_heads = set()
        for result in strategy_selections.values():
            all_heads.update(result['indices'].tolist())

        if len(all_heads) == 0:
            return 0.0

        # 计算overlap比例
        overlap_scores = []
        strategies = list(strategy_selections.keys())

        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                set1 = set(strategy_selections[strategies[i]]['indices'].tolist())
                set2 = set(strategy_selections[strategies[j]]['indices'].tolist())
                overlap = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
                overlap_scores.append(overlap)

        return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

    def adaptive_consensus_threshold(self, consistency, num_strategies):
        """
        根据策略一致性动态调整共识阈值
        """
        # 基础阈值
        base_threshold = 1.0 / 3.0

        # 高一致性时提高阈值，低一致性时降低阈值
        if consistency > 0.6:
            return min(0.5, base_threshold * (1 + consistency))
        elif consistency < 0.2:
            return max(0.15, base_threshold * consistency * 2)
        else:
            return base_threshold

    def compute_attention_complexity(self, image_attention):
        """
        计算attention pattern的复杂度
        """
        H, N = image_attention.shape

        # 复杂度指标1: 头间的多样性
        if H > 1:
            normalized = F.normalize(image_attention, p=2, dim=-1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            mask = ~torch.eye(H, dtype=torch.bool, device=image_attention.device)
            avg_similarity = similarity_matrix[mask].mean()
            head_diversity = 1 - avg_similarity
        else:
            head_diversity = 0.5

        # 复杂度指标2: 每个头内attention分布的复杂度
        attention_probs = torch.softmax(image_attention, dim=-1)
        entropies = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        avg_entropy = entropies.mean()
        max_entropy = torch.log(torch.tensor(float(N)))
        normalized_entropy = avg_entropy / max_entropy

        # 复杂度指标3: attention值的分布方差
        attention_variance = image_attention.var(dim=-1).mean()
        normalized_variance = attention_variance / (image_attention.mean() + 1e-8)

        # 综合复杂度
        complexity = 0.4 * head_diversity + 0.3 * normalized_entropy + 0.3 * normalized_variance.clamp(0, 1)
        return complexity.item()

    def complexity_aware_head_count(self, image_attention, head_weighted_votes, complexity):
        """
        基于attention复杂度感知的头数量确定
        """
        H, N = image_attention.shape

        # 基础方法：投票分布断点检测
        sorted_votes, _ = torch.sort(head_weighted_votes, descending=True)
        vote_gaps = sorted_votes[:-1] - sorted_votes[1:]
        if len(vote_gaps) > 0:
            mean_gap = vote_gaps.mean()
            std_gap = vote_gaps.std()
            significant_gaps = torch.where(vote_gaps > (mean_gap + 0.5 * std_gap))[0]
            gap_based = significant_gaps[0].item() + 1 if len(significant_gaps) > 0 else H // 4
        else:
            gap_based = H // 4

        # 复杂度调整
        if complexity > 0.7:
            # 高复杂度需要更多头
            complexity_adjusted = int(gap_based * (1.2 + 0.3 * complexity))
        elif complexity < 0.3:
            # 低复杂度需要较少头
            complexity_adjusted = int(gap_based * (0.7 + 0.6 * complexity))
        else:
            # 中等复杂度
            complexity_adjusted = gap_based

        return max(1, min(complexity_adjusted, H))

    def multi_level_diversity_selection(self, image_attention, available_heads, consensus_heads, count):
        """
        多层次差异化头选择：确保功能空间的全面覆盖
        """
        if count <= 0 or len(available_heads) == 0:
            return []

        available_heads = torch.tensor(available_heads, device=image_attention.device)
        available_attention = image_attention[available_heads]

        if len(consensus_heads) == 0:
            # 如果没有共识头，使用聚类选择多样化的头
            return self.clustering_based_selection(available_attention, available_heads, count)

        # === 层次1: 与共识头的差异性 ===
        consensus_heads_tensor = torch.tensor(consensus_heads, device=image_attention.device)
        consensus_attention = image_attention[consensus_heads_tensor]

        # 计算差异性分数
        consensus_norm = F.normalize(consensus_attention, p=2, dim=-1)
        available_norm = F.normalize(available_attention, p=2, dim=-1)
        similarity_matrix = torch.mm(available_norm, consensus_norm.t())
        avg_similarity = similarity_matrix.mean(dim=-1)
        diversity_from_consensus = 1 - avg_similarity

        # === 层次2: 候选头之间的多样性 ===
        selected_heads = []
        remaining_candidates = list(range(len(available_heads)))

        for _ in range(count):
            if not remaining_candidates:
                break

            if len(selected_heads) == 0:
                # 选择与共识头差异最大的头
                best_idx = diversity_from_consensus[remaining_candidates].argmax().item()
                selected_idx = remaining_candidates[best_idx]
            else:
                # 选择与已选头和共识头都差异最大的头
                selected_attention = available_attention[selected_heads]
                selected_norm = F.normalize(selected_attention, p=2, dim=-1)

                best_score = -float('inf')
                best_idx = 0

                for i, candidate_idx in enumerate(remaining_candidates):
                    candidate_attention = available_attention[candidate_idx:candidate_idx+1]
                    candidate_norm = F.normalize(candidate_attention, p=2, dim=-1)

                    # 与已选头的差异性
                    sim_to_selected = torch.mm(candidate_norm, selected_norm.t()).mean()
                    diversity_from_selected = 1 - sim_to_selected

                    # 综合分数：与共识头差异性 + 与已选头差异性
                    total_score = (diversity_from_consensus[candidate_idx] * 0.6 +
                                 diversity_from_selected * 0.4)

                    if total_score > best_score:
                        best_score = total_score
                        best_idx = i

                selected_idx = remaining_candidates[best_idx]

            selected_heads.append(selected_idx)
            remaining_candidates.remove(selected_idx)

        return available_heads[selected_heads].tolist()

    def clustering_based_selection(self, attention_data, head_indices, count):
        """
        基于聚类的多样化头选择
        """
        if len(attention_data) <= count:
            return head_indices.tolist()

        # 简单的基于距离的聚类选择
        selected = []
        remaining = list(range(len(attention_data)))

        # 选择第一个头（attention总和最高的）
        first_idx = attention_data.sum(dim=-1).argmax().item()
        selected.append(first_idx)
        remaining.remove(first_idx)

        # 逐一选择与已选头差异最大的头
        for _ in range(count - 1):
            if not remaining:
                break

            best_score = -float('inf')
            best_idx = 0

            for i, candidate_idx in enumerate(remaining):
                # 计算与已选头的最小距离
                min_similarity = float('inf')
                candidate_norm = F.normalize(attention_data[candidate_idx:candidate_idx+1], p=2, dim=-1)

                for selected_idx in selected:
                    selected_norm = F.normalize(attention_data[selected_idx:selected_idx+1], p=2, dim=-1)
                    similarity = F.cosine_similarity(candidate_norm, selected_norm, dim=-1).item()
                    min_similarity = min(min_similarity, similarity)

                diversity_score = 1 - min_similarity
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_idx = i

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return head_indices[selected].tolist()

    def compute_intelligent_weights(self, final_indices, consensus_heads, strategy_selections, image_attention):
        """
        基于头类型和质量的智能权重分配
        """
        weights = torch.ones(len(final_indices), device=image_attention.device)

        for i, head_idx in enumerate(final_indices):
            head_idx = head_idx.item()

            if head_idx in consensus_heads:
                # 共识头权重：基于其被策略选择的质量
                consensus_weight = 0.0
                vote_count = 0

                for strategy, result in strategy_selections.items():
                    if head_idx in result['indices']:
                        quality = result['quality']
                        consensus_weight += quality
                        vote_count += 1

                if vote_count > 0:
                    avg_quality = consensus_weight / vote_count
                    weights[i] = 1.0 + 0.5 * avg_quality  # 基础权重 + 质量奖励
                else:
                    weights[i] = 1.2  # 默认共识权重

            else:
                # 差异化头权重：基于其attention质量
                attention_quality = image_attention[head_idx].sum().item()
                max_attention = image_attention.sum(dim=-1).max().item()
                normalized_quality = attention_quality / (max_attention + 1e-8)
                weights[i] = 0.8 + 0.4 * normalized_quality  # 稍低的基础权重

        return torch.softmax(weights, dim=0)

    def determine_optimal_head_count(self, image_attention, head_vote_count):
        """
        动态确定最优头数量（范围1-32）
        基于attention pattern的内在特征，不使用后验知识
        """
        H, N = image_attention.shape

        # 方法1：基于投票分布的自然断点
        sorted_votes, _ = torch.sort(head_vote_count, descending=True)
        vote_gaps = sorted_votes[:-1] - sorted_votes[1:]
        if len(vote_gaps) > 0:
            mean_gap = vote_gaps.mean()
            std_gap = vote_gaps.std()
            significant_gap_threshold = mean_gap + 0.5 * std_gap
            significant_gaps = torch.where(vote_gaps > significant_gap_threshold)[0]
            if len(significant_gaps) > 0:
                gap_based_count = significant_gaps[0].item() + 1
            else:
                gap_based_count = max(8, H // 4)  # 默认值
        else:
            gap_based_count = max(8, H // 4)

        # 方法2：基于attention分布的有效性评估
        attention_std = image_attention.std(dim=-1)  # 每个头的attention方差
        attention_max = image_attention.max(dim=-1)[0]  # 每个头的最大attention
        effectiveness = attention_std * attention_max  # 结合方差和峰值

        effectiveness_sorted, _ = torch.sort(effectiveness, descending=True)
        effectiveness_threshold = effectiveness_sorted.mean() - 0.3 * effectiveness_sorted.std()
        effective_heads = (effectiveness > effectiveness_threshold).sum().item()

        # 方法3：基于累积质量贡献
        quality_scores = head_vote_count + 0.1 * effectiveness
        quality_sorted, _ = torch.sort(quality_scores, descending=True)
        quality_cumsum = torch.cumsum(quality_sorted, dim=0)
        quality_total = quality_cumsum[-1]
        # 选择能覆盖85%质量贡献的头数量
        contribution_85_idx = torch.where(quality_cumsum >= 0.85 * quality_total)[0]
        if len(contribution_85_idx) > 0:
            contribution_based_count = contribution_85_idx[0].item() + 1
        else:
            contribution_based_count = H // 2

        # 综合三种方法，取中位数
        candidates = [gap_based_count, effective_heads, contribution_based_count]
        optimal_count = sorted(candidates)[1]  # 中位数

        # 确保在合理范围内
        optimal_count = max(1, min(optimal_count, H))

        return optimal_count

    def select_diverse_heads(self, image_attention, available_heads, consensus_heads, count):
        """
        从可用头中选择最具差异性的头
        """
        if count <= 0 or len(available_heads) == 0:
            return []

        available_heads = torch.tensor(available_heads, device=image_attention.device)

        if len(consensus_heads) == 0:
            # 如果没有共识头，直接按attention质量选择
            available_attention = image_attention[available_heads]
            quality_scores = available_attention.sum(dim=-1)
            top_indices = quality_scores.topk(k=min(count, len(available_heads))).indices
            return available_heads[top_indices].tolist()

        # 计算与共识头的差异性
        consensus_heads_tensor = torch.tensor(consensus_heads, device=image_attention.device)
        consensus_attention = image_attention[consensus_heads_tensor]  # (C, N)
        available_attention = image_attention[available_heads]  # (A, N)

        # 计算每个可用头与共识头的平均相似度
        consensus_attention_norm = F.normalize(consensus_attention, p=2, dim=-1)
        available_attention_norm = F.normalize(available_attention, p=2, dim=-1)

        similarity_matrix = torch.mm(available_attention_norm, consensus_attention_norm.t())  # (A, C)
        avg_similarity = similarity_matrix.mean(dim=-1)  # (A,)

        # 差异性分数 = 1 - 相似性，同时考虑attention质量
        quality_scores = available_attention.sum(dim=-1)
        quality_scores_norm = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-8)

        diversity_scores = (1 - avg_similarity) * 0.7 + quality_scores_norm * 0.3

        # 选择差异性最高的头
        top_indices = diversity_scores.topk(k=min(count, len(available_heads))).indices
        return available_heads[top_indices].tolist()

    def compute_strategy_scores(self, image_attention, strategy):
        """计算特定策略的分数"""
        H, N = image_attention.shape

        if strategy == 'sparsity':
            attention_probs = torch.softmax(image_attention, dim=-1)
            sparsity = (attention_probs ** 2).sum(dim=-1)
            max_weight = attention_probs.max(dim=-1)[0]
            return 0.7 * sparsity + 0.3 * max_weight

        elif strategy == 'hierarchical':
            attention_probs = torch.softmax(image_attention, dim=-1)
            sparsity = (attention_probs ** 2).sum(dim=-1)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
            kl_div = F.kl_div(torch.log(attention_probs + 1e-8),
                             torch.ones_like(attention_probs) / N,
                             reduction='none').sum(dim=-1)
            return 0.4 * sparsity + 0.3 * (1 - entropy_norm) + 0.3 * kl_div

        elif strategy == 'graph_based':
            if H <= 1:
                return image_attention.sum(dim=-1)
            normalized = F.normalize(image_attention, p=2, dim=-1)
            similarity = torch.mm(normalized, normalized.t())
            centrality = similarity.sum(dim=-1)
            quality = image_attention.var(dim=-1)
            return 0.6 * centrality + 0.4 * quality

        elif strategy == 'top_k_sum':
            k = min(64, N // 4)
            topk_attention = image_attention.topk(k=k, dim=-1)[0]
            return topk_attention.sum(dim=-1)

        elif strategy == 'max_attention':
            return image_attention.max(dim=-1)[0]

        elif strategy == 'attention_range':
            return image_attention.max(dim=-1)[0] - image_attention.min(dim=-1)[0]

        elif strategy == 'multi_objective':
            attention_probs = torch.softmax(image_attention, dim=-1)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            sparsity = (attention_probs ** 2).sum(dim=-1)
            max_attn = attention_probs.max(dim=-1)[0]

            # 归一化并组合
            entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
            sparsity_norm = (sparsity - sparsity.min()) / (sparsity.max() - sparsity.min() + 1e-8)
            max_norm = (max_attn - max_attn.min()) / (max_attn.max() - max_attn.min() + 1e-8)
            return 0.4 * entropy_norm + 0.3 * sparsity_norm + 0.3 * max_norm

        else:
            return image_attention.sum(dim=-1)

    def determine_optimal_head_count(self, scores, preferred_range):
        """
        动态确定最优头数量，支持非传统固定值
        """
        min_heads, max_heads = preferred_range

        # 方法1: 基于分数分布的拐点检测
        sorted_scores, _ = torch.sort(scores, descending=True)

        if len(sorted_scores) > 3:
            # 计算相邻分数的差值
            gaps = sorted_scores[:-1] - sorted_scores[1:]

            # 寻找显著的质量下降点
            mean_gap = gaps.mean()
            std_gap = gaps.std()
            significant_gaps = gaps > (mean_gap + 0.8 * std_gap)

            if significant_gaps.any():
                # 找到第一个显著gap的位置
                first_gap_idx = significant_gaps.nonzero()[0].item()
                gap_based_count = first_gap_idx + 1
            else:
                gap_based_count = max_heads
        else:
            gap_based_count = len(sorted_scores)

        # 方法2: 基于质量阈值的筛选
        mean_score = sorted_scores.mean()
        std_score = sorted_scores.std()
        quality_threshold = mean_score - 0.3 * std_score  # 更宽松的阈值

        quality_based_count = (sorted_scores > quality_threshold).sum().item()

        # 方法3: 基于累积贡献度
        if len(sorted_scores) > 1:
            normalized_scores = sorted_scores / sorted_scores.sum()
            cumulative_contribution = torch.cumsum(normalized_scores, dim=0)
            # 找到累积贡献达到85%的点
            contribution_based_count = (cumulative_contribution <= 0.85).sum().item() + 1
        else:
            contribution_based_count = 1

        # 综合决策：取三种方法的中位数，确保在合理范围内
        candidates = [gap_based_count, quality_based_count, contribution_based_count]
        optimal_count = sorted(candidates)[1]  # 中位数

        # 确保在preferred_range范围内
        optimal_count = max(min_heads, min(optimal_count, max_heads))

        # 确保不超过可用头数
        optimal_count = min(optimal_count, len(scores))

        return optimal_count

    def evaluate_combination_quality(self, selected_attention, selected_scores):
        """
        评估头选择组合的整体质量
        """
        if len(selected_attention) == 0:
            return 0.0

        # 质量指标1: 多样性 - 选中heads之间的差异性
        if len(selected_attention) > 1:
            normalized = F.normalize(selected_attention, p=2, dim=-1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
            avg_similarity = similarity_matrix[mask].mean()
            diversity = 1 - avg_similarity
        else:
            diversity = 0.5

        # 质量指标2: 分数分布的合理性
        score_std = selected_scores.std()
        score_range = selected_scores.max() - selected_scores.min()
        score_quality = (score_std + score_range) / 2

        # 质量指标3: 注意力分布的有效性
        attention_variance = selected_attention.var(dim=-1).mean()
        attention_max = selected_attention.max(dim=-1)[0].mean()
        attention_quality = attention_variance * attention_max

        # 综合质量评分
        overall_quality = (
            0.3 * diversity +
            0.3 * score_quality +
            0.4 * attention_quality
        )

        return overall_quality.item()

    # ================== Forward函数 - 简化版本 ==================

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if seq_length > 1:
            visual_token_length = self.visual_token_length
            visual_token_num = 0

        # Head Selection - 读取配置
        if self.config.H is not None:
            self.H = self.config.H
            if self.debug_mode:
                print(f"Head_number updated to: {self.H}")

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                # 动态Head选择的核心逻辑
                if seq_length > 1:
                    visual_token_num += visual_token_length

                    if (decoder_layer.self_attn.layer_idx + 1) == self.K:
                        attn_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
                        attn_mask = _prepare_4d_causal_attention_mask(attn_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=True,
                            use_cache=use_cache,
                        )
                        hidden_states = layer_outputs[0]
                        last_attention = layer_outputs[1]

                        # 简化的attention获取：使用最后一个token对visual tokens的attention
                        image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

                        # 使用求同存异的共识-差异化选择策略
                        if self.enable_dynamic_selection:
                            visual_head_index, aggregated_attention = self.consensus_diversity_selection(image_attention)
                        else:
                            # Fallback to simple sum selection
                            head_attention = image_attention.sum(dim=-1)
                            visual_head_index = head_attention.topk(k=self.H).indices
                            aggregated_attention = image_attention[visual_head_index].mean(dim=0)

                        # 使用聚合后的attention分布选择visual tokens
                        image_attention = aggregated_attention

                        visual_token_length = self.R
                        visual_token_index = torch.topk(image_attention, k=self.R).indices # (R)
                        visual_token_index = torch.sort(visual_token_index).values # (R)

                        device = hidden_states.device
                        full_token_index = torch.cat((
                            torch.arange(self.system_prompt_length, device=device),
                            visual_token_index + self.system_prompt_length,
                            torch.arange(self.system_prompt_length+self.visual_token_length, seq_length, device=device),
                        )) # (T)

                        # get tokens by index
                        hidden_states = hidden_states[:,full_token_index]
                        position_ids = full_token_index.unsqueeze(0)

                        layer_outputs = (hidden_states, layer_outputs[2])

                    else:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                        )

                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if seq_length > 1:
            self.visual_token_num = visual_token_num / len(self.layers)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )