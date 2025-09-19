import torch
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


class FineFastVLlamaModelAblationA(LlamaModel):
    """
    消融研究A: 注意力头筛选方法比较
    固定使用最后一个query token，比较不同的头筛选策略效果
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
        self.H = 32  # 32(all) 24 16
        self.remove_sink = False

        # 消融研究A: 头筛选策略
        self.head_selection_strategy = getattr(config, 'head_selection_strategy', 'sum')
        print(f"🔧 Ablation A initialized with head_selection_strategy: {self.head_selection_strategy}")

    def update_head_selection_strategy(self, strategy):
        """动态更新头筛选策略"""
        self.head_selection_strategy = strategy
        print(f"🔧 Updated head_selection_strategy to: {strategy}")

    def get_last_query_attention(self, last_attention):
        """固定使用最后一个query token的注意力"""
        return last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

    def head_selection_sum(self, image_attention):
        """Baseline: 简单求和"""
        head_attention = image_attention.sum(dim=-1)  # (H,)
        visual_head_index = head_attention.topk(k=self.H).indices
        return visual_head_index

    def head_selection_variance(self, image_attention):
        """方法1: 基于注意力方差 - 选择注意力分布更集中的头"""
        head_variance = image_attention.var(dim=-1)  # (H,)
        visual_head_index = head_variance.topk(k=self.H).indices
        return visual_head_index

    def head_selection_entropy(self, image_attention):
        """方法2: 基于注意力熵 - 选择信息量适中的头"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)
        attention_entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)  # (H,)
        # 选择熵值较小的头（更集中的注意力）
        visual_head_index = attention_entropy.topk(k=self.H, largest=False).indices
        return visual_head_index

    def head_selection_max_attention(self, image_attention):
        """方法3: 基于最大注意力值 - 选择有强烈关注点的头"""
        max_attention = image_attention.max(dim=-1)[0]  # (H,)
        visual_head_index = max_attention.topk(k=self.H).indices
        return visual_head_index

    def head_selection_attention_range(self, image_attention):
        """方法4: 基于注意力范围 - 选择注意力分布范围大的头"""
        attention_range = image_attention.max(dim=-1)[0] - image_attention.min(dim=-1)[0]  # (H,)
        visual_head_index = attention_range.topk(k=self.H).indices
        return visual_head_index

    def head_selection_sparsity(self, image_attention):
        """方法5: 基于注意力稀疏性 - 选择注意力更稀疏的头"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)
        # 计算L2范数作为稀疏性指标
        sparsity_score = (attention_probs ** 2).sum(dim=-1)  # (H,)
        visual_head_index = sparsity_score.topk(k=self.H).indices
        return visual_head_index

    def head_selection_top_k_sum(self, image_attention):
        """方法6: 基于top-k注意力和 - 选择top-k注意力值总和最大的头"""
        k = min(10, image_attention.size(-1))  # 选择top-10或更少
        topk_attention = image_attention.topk(k=k, dim=-1)[0]  # (H, k)
        topk_sum = topk_attention.sum(dim=-1)  # (H,)
        visual_head_index = topk_sum.topk(k=self.H).indices
        return visual_head_index

    def head_selection_weighted_quality(self, image_attention):
        """方法7: 加权质量评分 - 综合多个指标"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)

        # 指标1: 最大注意力值（集中度）
        max_attention = attention_probs.max(dim=-1)[0]  # (H,)

        # 指标2: 注意力分布的尖锐度
        attention_sharpness = (attention_probs ** 2).sum(dim=-1)  # (H,)

        # 指标3: 有效注意力范围
        significant_tokens = (attention_probs > 0.01).sum(dim=-1).float()  # (H,)
        attention_coverage = 1.0 / (significant_tokens + 1e-8)

        # 综合质量分数
        quality_score = (max_attention * 0.4 +
                        attention_sharpness * 0.4 +
                        attention_coverage * 0.2)
        visual_head_index = quality_score.topk(k=self.H).indices
        return visual_head_index

    def head_selection_gini_coefficient(self, image_attention):
        """方法8: 基于基尼系数 - 选择注意力分布不均匀的头"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)

        # 计算基尼系数
        sorted_attention, _ = torch.sort(attention_probs, dim=-1)  # (H, N)
        n = sorted_attention.size(-1)
        index = torch.arange(1, n + 1, device=sorted_attention.device).float()
        gini = ((2 * index - n - 1) * sorted_attention).sum(dim=-1) / (n * sorted_attention.sum(dim=-1))

        visual_head_index = gini.topk(k=self.H).indices
        return visual_head_index

    def head_selection_multi_objective(self, image_attention):
        """方法9: 多目标协同优化 - 结合信息丰富度、多样性和稀疏性"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)
        H, N = attention_probs.shape

        # 目标1: 信息丰富度 (结合熵和峰度)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        mean_attn = attention_probs.mean(dim=-1, keepdim=True)
        variance = ((attention_probs - mean_attn) ** 2).mean(dim=-1)
        kurtosis = ((attention_probs - mean_attn) ** 4).mean(dim=-1) / (variance ** 2 + 1e-8)

        # 归一化
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        kurtosis_norm = (kurtosis - kurtosis.min()) / (kurtosis.max() - kurtosis.min() + 1e-8)
        info_richness = 0.7 * entropy_norm + 0.3 * kurtosis_norm

        # 目标2: 头多样性 (基于相互相似度)
        attention_norm = torch.nn.functional.normalize(attention_probs, p=2, dim=-1)
        similarity_matrix = torch.mm(attention_norm, attention_norm.t())
        avg_similarity = (similarity_matrix.sum(dim=-1) - 1) / (H - 1)  # 排除自身
        diversity_score = 1 / (avg_similarity + 1e-8)

        # 目标3: 稀疏性
        sparsity_score = (attention_probs ** 2).sum(dim=-1)

        # 归一化所有目标
        diversity_norm = (diversity_score - diversity_score.min()) / (diversity_score.max() - diversity_score.min() + 1e-8)
        sparsity_norm = (sparsity_score - sparsity_score.min()) / (sparsity_score.max() - sparsity_score.min() + 1e-8)

        # 多目标融合 (可调权重)
        alpha, beta, gamma = 0.4, 0.3, 0.3
        composite_score = alpha * info_richness + beta * diversity_norm + gamma * sparsity_norm

        visual_head_index = composite_score.topk(k=self.H).indices
        return visual_head_index

    def head_selection_graph_based(self, image_attention):
        """方法10: 基于图结构的头选择 - 考虑头之间的关系网络"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)
        H, N = attention_probs.shape

        # 1. 构建节点特征 (每个头的5维特征)
        sparsity = (attention_probs ** 2).sum(dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        max_attention = attention_probs.max(dim=-1)[0]
        variance = attention_probs.var(dim=-1)
        mean_attn = attention_probs.mean(dim=-1, keepdim=True)
        effective_range = (attention_probs > mean_attn).float().mean(dim=-1)

        # 节点特征矩阵 (H, 5)
        node_features = torch.stack([sparsity, entropy, max_attention, variance, effective_range], dim=-1)

        # 2. 构建邻接矩阵 (基于注意力模式相似度)
        attention_norm = torch.nn.functional.normalize(attention_probs, p=2, dim=-1)
        similarity_matrix = torch.mm(attention_norm, attention_norm.t())
        threshold = 0.1  # 相似度阈值
        adjacency_matrix = (similarity_matrix > threshold).float()
        adjacency_matrix.fill_diagonal_(0)  # 移除自环

        # 归一化邻接矩阵
        row_sum = adjacency_matrix.sum(dim=-1, keepdim=True)
        adjacency_matrix = adjacency_matrix / (row_sum + 1e-8)

        # 3. 简化的图卷积 (消息传递)
        # 第一层
        messages_1 = torch.mm(adjacency_matrix, node_features)
        node_features_1 = torch.relu(node_features + messages_1)

        # 第二层
        messages_2 = torch.mm(adjacency_matrix, node_features_1)
        node_features_2 = torch.relu(node_features_1 + messages_2)

        # 4. 聚合特征并计算最终得分
        final_scores = node_features_2.mean(dim=-1)  # 简单平均

        visual_head_index = final_scores.topk(k=self.H).indices
        return visual_head_index

    def head_selection_hierarchical(self, image_attention):
        """方法11: 分层选择策略 - 粗选+精选的两阶段方法"""
        attention_probs = torch.softmax(image_attention, dim=-1)  # (H, N)
        H, N = attention_probs.shape

        # 第一阶段: 粗选 (保留更多候选)
        coarse_k = min(H, max(self.H * 2, 16))  # 粗选保留2倍目标数量

        # 粗选指标: 简单的稀疏性+熵组合
        sparsity = (attention_probs ** 2).sum(dim=-1)
        entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        coarse_scores = 0.6 * sparsity + 0.4 * (1 - entropy_norm)

        # 获取粗选候选
        coarse_indices = coarse_scores.topk(k=coarse_k).indices
        coarse_attention = attention_probs[coarse_indices]

        # 第二阶段: 精选 (复杂指标)
        # 1. 信息理论指标
        fine_entropy = -(coarse_attention * torch.log(coarse_attention + 1e-8)).sum(dim=-1)

        # 2. KL散度 (与均匀分布的距离)
        uniform_dist = torch.ones_like(coarse_attention) / N
        kl_div = torch.nn.functional.kl_div(
            torch.log(coarse_attention + 1e-8), uniform_dist, reduction='none'
        ).sum(dim=-1)

        # 3. 互补性 (与其他选中头的差异)
        pairwise_distances = torch.cdist(coarse_attention, coarse_attention, p=2)
        avg_distance = pairwise_distances.mean(dim=-1)

        # 4. 稳定性指标
        stability = 1 / (coarse_attention.std(dim=-1) + 1e-8)

        # 归一化并组合所有指标
        metrics = [fine_entropy, kl_div, avg_distance, stability]
        normalized_metrics = []
        for metric in metrics:
            metric_norm = (metric - metric.min()) / (metric.max() - metric.min() + 1e-8)
            normalized_metrics.append(metric_norm)

        # 精选得分
        weights = [0.3, 0.25, 0.25, 0.2]
        fine_scores = sum(w * m for w, m in zip(weights, normalized_metrics))

        # 获取最终选择
        fine_indices = fine_scores.topk(k=self.H).indices
        visual_head_index = coarse_indices[fine_indices]

        return visual_head_index

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
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
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
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if seq_length > 1:
            visual_token_length = self.visual_token_length
            visual_token_num = 0

        # Head Selection
        if self.config.H is not None:
            self.H = self.config.H
            print(f"Head_number:{self.H}")

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

                # 消融研究A: 头筛选方法比较
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

                        # 固定使用最后一个query token
                        image_attention = self.get_last_query_attention(last_attention)  # (H, N)

                        # 根据策略选择注意力头 - 动态读取config
                        current_strategy = getattr(self.config, 'head_selection_strategy', self.head_selection_strategy)
                        print(f"🔧 Using head selection strategy: {current_strategy}")
                        if current_strategy == 'sum':
                            visual_head_index = self.head_selection_sum(image_attention)
                        elif current_strategy == 'variance':
                            visual_head_index = self.head_selection_variance(image_attention)
                        elif current_strategy == 'entropy':
                            visual_head_index = self.head_selection_entropy(image_attention)
                        elif current_strategy == 'max_attention':
                            visual_head_index = self.head_selection_max_attention(image_attention)
                        elif current_strategy == 'attention_range':
                            visual_head_index = self.head_selection_attention_range(image_attention)
                        elif current_strategy == 'sparsity':
                            visual_head_index = self.head_selection_sparsity(image_attention)
                        elif current_strategy == 'top_k_sum':
                            visual_head_index = self.head_selection_top_k_sum(image_attention)
                        elif current_strategy == 'weighted_quality':
                            visual_head_index = self.head_selection_weighted_quality(image_attention)
                        elif current_strategy == 'gini_coefficient':
                            visual_head_index = self.head_selection_gini_coefficient(image_attention)
                        elif current_strategy == 'multi_objective':
                            visual_head_index = self.head_selection_multi_objective(image_attention)
                        elif current_strategy == 'graph_based':
                            visual_head_index = self.head_selection_graph_based(image_attention)
                        elif current_strategy == 'hierarchical':
                            visual_head_index = self.head_selection_hierarchical(image_attention)
                        else:
                            # 默认使用sum方法
                            visual_head_index = self.head_selection_sum(image_attention)

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

        # add hidden states from the last decoder layer
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