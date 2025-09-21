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

    def select_heads_by_strategy(self, image_attention):
        """
        简化的头选择策略：基于配置的策略选择方法

        Args:
            image_attention: (H, N) - 每个head对visual tokens的attention

        Returns:
            visual_head_index: 选中的head索引
            aggregated_attention: 聚合后的attention分布
        """
        H, N = image_attention.shape

        if self.head_selection_strategy == 'sum':
            # 原始FastV方法：选择attention总和最高的heads
            head_scores = image_attention.sum(dim=-1)

        elif self.head_selection_strategy == 'variance':
            # 选择attention分布方差最大的heads (信息丰富度)
            head_scores = image_attention.var(dim=-1)

        elif self.head_selection_strategy == 'entropy':
            # 选择attention熵适中的heads
            attention_probs = torch.softmax(image_attention, dim=-1)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            # 选择熵在中等范围的heads
            mean_entropy = entropy.mean()
            head_scores = -(entropy - mean_entropy).abs()

        elif self.head_selection_strategy == 'max_attention':
            # 选择最大attention值最高的heads
            head_scores = image_attention.max(dim=-1)[0]

        elif self.head_selection_strategy == 'attention_range':
            # 选择attention值范围最大的heads
            head_scores = image_attention.max(dim=-1)[0] - image_attention.min(dim=-1)[0]

        elif self.head_selection_strategy == 'sparsity':
            # 选择稀疏性适中的heads
            attention_probs = torch.softmax(image_attention, dim=-1)
            sparsity = (attention_probs ** 2).sum(dim=-1)
            head_scores = sparsity

        elif self.head_selection_strategy == 'top_k_sum':
            # 选择top-k attention总和最高的heads
            k = min(64, N // 4)
            topk_attention = image_attention.topk(k=k, dim=-1)[0]
            head_scores = topk_attention.sum(dim=-1)

        elif self.head_selection_strategy == 'weighted_quality':
            # 结合峰值和方差的质量分数
            max_attn = image_attention.max(dim=-1)[0]
            var_attn = image_attention.var(dim=-1)
            head_scores = max_attn * var_attn

        elif self.head_selection_strategy == 'gini_coefficient':
            # 基于基尼系数选择heads
            sorted_attn = torch.sort(image_attention, dim=-1)[0]
            n = sorted_attn.shape[-1]
            index = torch.arange(1, n + 1, device=sorted_attn.device).float()
            gini = (2 * (sorted_attn * index).sum(dim=-1)) / (n * sorted_attn.sum(dim=-1)) - (n + 1) / n
            head_scores = gini

        else:
            # 默认使用sum策略
            head_scores = image_attention.sum(dim=-1)

        # 选择top-H heads
        visual_head_index = head_scores.topk(k=self.H).indices

        # 聚合选中heads的attention - 使用加权平均
        selected_attention = image_attention[visual_head_index]  # (H, N)
        selected_scores = head_scores[visual_head_index]  # (H,)

        # 计算权重 (归一化后的分数)
        weights = torch.softmax(selected_scores, dim=0)  # (H,)
        aggregated_attention = (selected_attention * weights.unsqueeze(-1)).sum(dim=0)  # (N,)

        return visual_head_index, aggregated_attention

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

                        # 使用简化的头选择策略
                        if self.enable_dynamic_selection:
                            visual_head_index, aggregated_attention = self.select_heads_by_strategy(image_attention)
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