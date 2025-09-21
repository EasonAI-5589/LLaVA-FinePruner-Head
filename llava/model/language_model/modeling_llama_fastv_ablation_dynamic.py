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
    动态Head选择策略: 层次化智能选择
    Step 1: 粗筛选 - 检测视觉敏感头
    Step 2: 策略判断 - 智能选择筛选策略
    Step 3: 精筛选 - 应用策略精细筛选
    Step 4: 自适应数量 - 动态调整头数量
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
        self.H = 32  # 基础目标头数量，将动态调整
        self.remove_sink = False

        # 动态选择配置
        self.enable_dynamic_selection = getattr(config, 'enable_dynamic_selection', True)
        self.min_heads = getattr(config, 'min_heads', 8)
        self.max_heads = getattr(config, 'max_heads', 24)
        self.debug_mode = False  # 简化，默认关闭debug

        print(f"🔧 Dynamic Head Selection initialized: enabled={self.enable_dynamic_selection}")
        print(f"   Target heads: {self.H}, Range: [{self.min_heads}, {self.max_heads}]")

    def get_last_query_attention(self, last_attention):
        """固定使用最后一个query token的注意力"""
        return last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

    # ================== Step 1: 粗筛选 - 视觉敏感度检测 ==================

    def dynamic_visual_head_detection(self, image_attention):
        """
        粗筛选：检测对视觉信息敏感的注意力头

        输入: image_attention (32, 576)
        输出: visual_head_indices, head_sensitivity_scores
        """
        H, N = image_attention.shape  # (32, 576)

        # 指标1: 总注意力强度 - 衡量头对视觉token的总体关注度
        total_attention = image_attention.sum(dim=-1)  # (32,)

        # 指标2: 注意力方差 - 衡量注意力分布的不均匀性
        attention_variance = image_attention.var(dim=-1)  # (32,)

        # 指标3: 最大注意力值 - 衡量是否有强烈的关注点
        max_attention = image_attention.max(dim=-1)[0]  # (32,)

        # 指标4: 有效关注范围 - 衡量关注的visual token数量
        mean_attention = image_attention.mean(dim=-1, keepdim=True)  # (32, 1)
        effective_tokens = (image_attention > mean_attention).float().sum(dim=-1)  # (32,)

        # 指标5: 注意力集中度 - top-10%的注意力占比
        topk_size = max(1, N // 10)  # top 10%
        topk_attention = image_attention.topk(k=topk_size, dim=-1)[0].sum(dim=-1)  # (32,)
        concentration = topk_attention / (total_attention + 1e-8)  # (32,)

        # 归一化各指标到[0,1]
        def normalize_score(score):
            return (score - score.min()) / (score.max() - score.min() + 1e-8)

        total_norm = normalize_score(total_attention)
        variance_norm = normalize_score(attention_variance)
        max_norm = normalize_score(max_attention)
        effective_norm = normalize_score(effective_tokens)
        concentration_norm = normalize_score(concentration)

        # 综合视觉敏感度评分 (权重可调)
        visual_sensitivity = (
            0.25 * total_norm +           # 总关注度
            0.20 * variance_norm +        # 分布不均匀性
            0.25 * max_norm +             # peak关注度
            0.15 * effective_norm +       # 关注广度
            0.15 * concentration_norm     # 集中度
        )

        # 动态阈值策略
        # 方法1: 统计阈值
        mean_sens = visual_sensitivity.mean()
        std_sens = visual_sensitivity.std()
        threshold_1 = mean_sens + 0.5 * std_sens

        # 方法2: 分位数阈值
        threshold_2 = torch.quantile(visual_sensitivity, 0.6)  # 保留top 40%

        # 方法3: 动态gap检测
        sorted_sens, _ = torch.sort(visual_sensitivity, descending=True)
        if len(sorted_sens) > 1:
            gaps = sorted_sens[:-1] - sorted_sens[1:]  # 计算相邻分数的gap
            max_gap_idx = gaps.argmax().item()
            threshold_3 = sorted_sens[max_gap_idx + 1]
        else:
            threshold_3 = threshold_1

        # 选择最合适的阈值
        final_threshold = max(threshold_1, threshold_2, threshold_3)

        # 获取超过阈值的头
        visual_heads_mask = visual_sensitivity > final_threshold
        visual_head_indices = torch.where(visual_heads_mask)[0]

        # 保证最少数量 (避免选择过少)
        if len(visual_head_indices) < self.min_heads:
            visual_head_indices = visual_sensitivity.topk(k=self.min_heads).indices

        # 保证最多数量 (避免选择过多)
        if len(visual_head_indices) > self.max_heads:
            selected_scores = visual_sensitivity[visual_head_indices]
            top_indices = selected_scores.topk(k=self.max_heads).indices
            visual_head_indices = visual_head_indices[top_indices]

        if self.debug_mode:
            print(f"   Visual sensitivity stats: mean={mean_sens:.3f}, std={std_sens:.3f}")
            print(f"   Thresholds: stat={threshold_1:.3f}, quantile={threshold_2:.3f}, gap={threshold_3:.3f}")
            print(f"   Selected {len(visual_head_indices)}/32 visual-sensitive heads")

        return visual_head_indices, visual_sensitivity[visual_head_indices]

    # ================== Step 2: 策略判断 - 智能策略选择 ==================

    def adaptive_strategy_selection(self, image_attention, visual_head_indices):
        """
        策略判断：分析注意力模式，选择最适合的精筛策略

        输入: image_attention (32, 576), visual_head_indices (M,)
        输出: best_strategy (str)
        """
        selected_attention = image_attention[visual_head_indices]  # (M, 576)
        M, N = selected_attention.shape

        # 特征1: 注意力熵分析
        attention_probs = torch.softmax(selected_attention, dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)  # (M,)
        avg_entropy = entropy.mean().item()

        # 特征2: 稀疏性分析
        sparsity = (attention_probs ** 2).sum(dim=-1)  # (M,)
        avg_sparsity = sparsity.mean().item()

        # 特征3: 头多样性分析
        def compute_head_diversity(attention_matrix):
            normalized = F.normalize(attention_matrix, p=2, dim=-1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            # 去掉对角线自相似
            if similarity_matrix.size(0) > 1:
                mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
                avg_similarity = similarity_matrix[mask].mean().item()
                return 1 - avg_similarity  # 多样性 = 1 - 相似性
            else:
                return 0.5  # 单个头时返回中性值

        diversity = compute_head_diversity(selected_attention)

        # 特征4: 注意力集中度
        max_attention = selected_attention.max(dim=-1)[0]  # (M,)
        avg_concentration = max_attention.mean().item()

        # 特征5: 注意力分布稳定性
        attention_std = selected_attention.std(dim=-1).mean().item()

        if self.debug_mode:
            print(f"   Attention Analysis: entropy={avg_entropy:.3f}, sparsity={avg_sparsity:.3f}, "
                  f"diversity={diversity:.3f}, concentration={avg_concentration:.3f}, stability={attention_std:.3f}")

        # 策略决策树 (基于实验数据优化)
        if avg_entropy > 2.5 and diversity > 0.7:
            # 高熵 + 高多样性 → 复杂多目标场景
            if avg_concentration > 0.3:
                strategy = 'graph_based'      # 复杂关系，用图方法
            else:
                strategy = 'multi_objective'  # 多目标优化

        elif avg_sparsity > 0.8 and avg_concentration > 0.25:
            # 高稀疏 + 高集中 → 明确目标场景
            strategy = 'sparsity'  # 实验显示sparsity在这种情况下最优

        elif avg_entropy < 1.8 and attention_std < 0.1:
            # 低熵 + 稳定 → 简单场景
            strategy = 'hierarchical'  # 简单场景用分层方法

        elif diversity < 0.3:
            # 低多样性 → 头之间相似，需要区分
            strategy = 'attention_range'  # 用range来区分

        elif avg_concentration < 0.15:
            # 低集中度 → 注意力分散
            strategy = 'top_k_sum'  # 聚焦top-k

        else:
            # 中等复杂度 → 平衡策略
            strategy = 'multi_objective'

        if self.debug_mode:
            print(f"   Selected strategy: {strategy}")
        return strategy

    # ================== Step 3: 精筛选 - 策略应用 ==================

    def apply_refined_selection(self, selected_attention, strategy, target_heads):
        """
        精筛选：在候选头中应用特定策略

        输入: selected_attention (M, 576), strategy (str), target_heads (int)
        输出: refined_indices (相对于selected_attention的索引)
        """
        # 确保target_heads不超过可用头数
        target_heads = min(target_heads, selected_attention.size(0))

        # 根据策略计算精细分数
        if strategy == 'sparsity':
            scores = self.compute_sparsity_scores(selected_attention)
        elif strategy == 'graph_based':
            scores = self.compute_graph_scores(selected_attention)
        elif strategy == 'hierarchical':
            scores = self.compute_hierarchical_scores(selected_attention)
        elif strategy == 'multi_objective':
            scores = self.compute_multi_objective_scores(selected_attention)
        elif strategy == 'attention_range':
            scores = self.compute_attention_range_scores(selected_attention)
        elif strategy == 'top_k_sum':
            scores = self.compute_top_k_sum_scores(selected_attention)
        elif strategy == 'max_attention':
            scores = self.compute_max_attention_scores(selected_attention)
        else:
            # fallback
            scores = selected_attention.sum(dim=-1)

        # 选择top-k
        refined_indices = scores.topk(k=target_heads).indices
        return refined_indices

    def compute_sparsity_scores(self, attention):
        """计算稀疏性分数 - 优化版本"""
        attention_probs = torch.softmax(attention, dim=-1)
        # L2范数 + 最大值权重
        l2_norm = (attention_probs ** 2).sum(dim=-1)
        max_weight = attention_probs.max(dim=-1)[0]
        return 0.7 * l2_norm + 0.3 * max_weight

    def compute_graph_scores(self, attention):
        """计算图分数 - 简化版本"""
        if attention.size(0) <= 1:
            return attention.sum(dim=-1)

        # 计算节点中心性
        normalized = F.normalize(attention, p=2, dim=-1)
        similarity = torch.mm(normalized, normalized.t())
        centrality = similarity.sum(dim=-1)  # 度中心性

        # 结合注意力质量
        quality = attention.var(dim=-1)  # 注意力方差
        return 0.6 * centrality + 0.4 * quality

    def compute_hierarchical_scores(self, attention):
        """计算分层分数"""
        attention_probs = torch.softmax(attention, dim=-1)

        # 第一层: 基础指标
        sparsity = (attention_probs ** 2).sum(dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

        # 第二层: 复合指标
        kl_div = F.kl_div(torch.log(attention_probs + 1e-8),
                          torch.ones_like(attention_probs) / attention_probs.size(-1),
                          reduction='none').sum(dim=-1)

        return 0.4 * sparsity + 0.3 * (1 - entropy_norm) + 0.3 * kl_div

    def compute_multi_objective_scores(self, attention):
        """计算多目标分数"""
        attention_probs = torch.softmax(attention, dim=-1)

        # 目标1: 信息丰富度
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)

        # 目标2: 稀疏性
        sparsity = (attention_probs ** 2).sum(dim=-1)

        # 目标3: 集中度
        max_attn = attention_probs.max(dim=-1)[0]

        # 归一化并组合
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        sparsity_norm = (sparsity - sparsity.min()) / (sparsity.max() - sparsity.min() + 1e-8)
        max_norm = (max_attn - max_attn.min()) / (max_attn.max() - max_attn.min() + 1e-8)

        return 0.4 * entropy_norm + 0.3 * sparsity_norm + 0.3 * max_norm

    def compute_attention_range_scores(self, attention):
        """计算注意力范围分数"""
        return attention.max(dim=-1)[0] - attention.min(dim=-1)[0]

    def compute_top_k_sum_scores(self, attention):
        """计算top-k和分数"""
        k = min(10, attention.size(-1))
        topk_attention = attention.topk(k=k, dim=-1)[0]
        return topk_attention.sum(dim=-1)

    def compute_max_attention_scores(self, attention):
        """计算最大注意力分数"""
        return attention.max(dim=-1)[0]

    # ================== Step 4: 自适应数量调整 ==================

    def adaptive_head_count(self, refined_scores, base_target):
        """
        自适应头数量：根据质量分布动态调整数量
        """
        if len(refined_scores) <= 1:
            return len(refined_scores)

        # 分析分数分布
        sorted_scores, _ = torch.sort(refined_scores, descending=True)

        # 方法1: Gap-based stopping
        gaps = sorted_scores[:-1] - sorted_scores[1:]
        mean_gap = gaps.mean()
        std_gap = gaps.std()

        # 找到显著gap
        significant_gaps = gaps > (mean_gap + std_gap)
        if significant_gaps.any():
            stop_idx = significant_gaps.nonzero()[0].item() + 1
            adaptive_count = min(stop_idx, base_target)
        else:
            adaptive_count = base_target

        # 方法2: Quality threshold
        mean_score = sorted_scores.mean()
        std_score = sorted_scores.std()
        threshold = mean_score - 0.5 * std_score

        quality_count = (sorted_scores > threshold).sum().item()

        # 综合决策
        final_count = min(
            max(adaptive_count, quality_count // 2),  # 至少保证质量
            base_target,  # 不超过目标
            len(sorted_scores)  # 不超过可用数量
        )

        return max(final_count, 1)  # 至少保证1个头

    # ================== 主要的动态选择流程 ==================

    def hierarchical_dynamic_selection(self, image_attention):
        """
        完整的层次化动态选择流程
        """
        if self.debug_mode:
            print(f"🚀 Starting hierarchical dynamic selection...")

        # Step 1: 粗筛选
        visual_head_indices, sensitivity_scores = self.dynamic_visual_head_detection(image_attention)

        # Step 2: 策略判断
        strategy = self.adaptive_strategy_selection(image_attention, visual_head_indices)

        # Step 3: 精筛选
        selected_attention = image_attention[visual_head_indices]
        refined_indices = self.apply_refined_selection(selected_attention, strategy, self.H)

        # Step 4: 自适应数量
        refined_scores = sensitivity_scores[refined_indices]  # 使用之前的sensitivity scores
        final_count = self.adaptive_head_count(refined_scores, self.H)

        # 获取最终头索引
        final_refined_indices = refined_indices[:final_count]
        final_head_indices = visual_head_indices[final_refined_indices]

        if self.debug_mode:
            print(f"✅ Final result: {len(final_head_indices)} heads using {strategy} strategy")

        return final_head_indices, strategy

    # ================== Forward函数 - 保持与原代码兼容 ==================

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

                        # 获取最后一个query token的注意力
                        image_attention = self.get_last_query_attention(last_attention)  # (H, N)

                        # 使用动态选择策略
                        if self.enable_dynamic_selection:
                            visual_head_index, selected_strategy = self.hierarchical_dynamic_selection(image_attention)
                            if self.debug_mode:
                                print(f"🎯 Dynamic selection: {len(visual_head_index)} heads, strategy: {selected_strategy}")
                        else:
                            # Fallback to simple sum selection
                            head_attention = image_attention.sum(dim=-1)
                            visual_head_index = head_attention.topk(k=self.H).indices
                            if self.debug_mode:
                                print(f"🔧 Fallback: using simple sum selection with {self.H} heads")

                        # 使用选中的头计算visual token重要性
                        image_attention = image_attention[visual_head_index].mean(dim=0)  # (N)

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