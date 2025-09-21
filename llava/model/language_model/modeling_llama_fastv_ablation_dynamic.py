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

    def adaptive_strategy_and_count_selection(self, image_attention):
        """
        基于预实验结果的自适应策略和头数量选择

        根据token数量自适应选择最优策略，并动态确定头数量（支持非固定值）
        """
        H, N = image_attention.shape

        # 第一步：基于token数量确定候选策略（基于预实验数据）
        if self.R >= 166:  # 对应192 tokens
            # 高资源场景：sparsity和hierarchical表现最佳
            candidate_strategies = ['sparsity', 'hierarchical']
            preferred_head_range = (14, 26)  # 扩展范围，支持非固定值
        elif self.R >= 98:  # 对应128 tokens
            # 中等资源场景：多策略混合
            candidate_strategies = ['sparsity', 'hierarchical', 'top_k_sum', 'max_attention']
            preferred_head_range = (8, 26)
        else:  # 64 tokens及以下
            # 低资源场景：graph_based表现最佳
            candidate_strategies = ['graph_based', 'multi_objective', 'attention_range']
            preferred_head_range = (6, 18)

        # 第二步：评估每个候选策略的质量
        strategy_results = {}
        for strategy in candidate_strategies:
            scores = self.compute_strategy_scores(image_attention, strategy)
            optimal_count = self.determine_optimal_head_count(scores, preferred_head_range)

            # 选择top heads
            selected_indices = scores.topk(k=optimal_count).indices
            selected_attention = image_attention[selected_indices]

            # 评估这个组合的质量
            quality_score = self.evaluate_combination_quality(selected_attention, scores[selected_indices])

            strategy_results[strategy] = {
                'indices': selected_indices,
                'count': optimal_count,
                'quality': quality_score,
                'scores': scores[selected_indices]
            }

        # 第三步：选择最优策略组合
        best_strategy = max(strategy_results.keys(), key=lambda x: strategy_results[x]['quality'])
        best_result = strategy_results[best_strategy]

        # 第四步：聚合选中heads的attention
        selected_attention = image_attention[best_result['indices']]
        weights = torch.softmax(best_result['scores'], dim=0)
        aggregated_attention = (selected_attention * weights.unsqueeze(-1)).sum(dim=0)

        if self.debug_mode:
            print(f"🎯 Adaptive selection: {best_result['count']} heads using {best_strategy} strategy")
            print(f"   Quality score: {best_result['quality']:.3f}")

        return best_result['indices'], aggregated_attention

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

                        # 使用基于预实验数据的自适应选择策略
                        if self.enable_dynamic_selection:
                            visual_head_index, aggregated_attention = self.adaptive_strategy_and_count_selection(image_attention)
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