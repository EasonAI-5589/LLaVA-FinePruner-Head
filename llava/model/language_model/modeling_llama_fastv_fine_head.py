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


class FineFastVLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
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

        # 头选择策略配置
        self.head_selection_strategy = getattr(config, 'head_selection_strategy', 'sum')
        print(f"🔧 Head selection strategy: {self.head_selection_strategy}")

    def select_heads_by_strategy(self, image_attention):
        """
        根据配置的策略选择attention heads

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
            # 选择attention熵适中的heads (既不太分散也不太集中)
            attention_probs = torch.softmax(image_attention, dim=-1)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            # 选择熵在中等范围的heads
            mean_entropy = entropy.mean()
            head_scores = -(entropy - mean_entropy).abs()  # 距离平均熵越近分数越高

        elif self.head_selection_strategy == 'max_attention':
            # 选择最大attention值最高的heads
            head_scores = image_attention.max(dim=-1)[0]

        elif self.head_selection_strategy == 'attention_range':
            # 选择attention值范围最大的heads
            head_scores = image_attention.max(dim=-1)[0] - image_attention.min(dim=-1)[0]

        elif self.head_selection_strategy == 'sparsity':
            # 选择稀疏性适中的heads
            attention_probs = torch.softmax(image_attention, dim=-1)
            sparsity = (attention_probs ** 2).sum(dim=-1)  # L2 norm (higher = more sparse)
            head_scores = sparsity

        elif self.head_selection_strategy == 'top_k_sum':
            # 选择top-k attention总和最高的heads
            k = min(64, N // 4)  # top 25%
            topk_attention = image_attention.topk(k=k, dim=-1)[0]
            head_scores = topk_attention.sum(dim=-1)

        elif self.head_selection_strategy == 'weighted_quality':
            # 结合峰值和方差的质量分数
            max_attn = image_attention.max(dim=-1)[0]
            var_attn = image_attention.var(dim=-1)
            head_scores = max_attn * var_attn

        elif self.head_selection_strategy == 'gini_coefficient':
            # 基于基尼系数选择heads (衡量不平等性)
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

        # 存储每一层的注意力分数
        layer_attention_scores = []

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

                # FastV
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
                        image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+visual_token_length]


                        # 简化的头选择策略 - 基于配置的策略选择方法
                        visual_head_index, final_image_attention = self.select_heads_by_strategy(image_attention)
                        
                        # 两种筛选方法选择
                        use_center_region = False  # True: 基于空间位置选择, False: 基于attention值选择
                        
                        if use_center_region:
                            # 方法1: 基于空间位置选择 - 从中心向两边扩散选择self.R个visual token
                            N = image_attention.shape[-1]  # visual token总数 (576)
                            center_pos = N // 2  # 中心位置 (288)
                            
                            # 计算self.R个token的起始和结束位置，以中心为基准向两边扩散
                            target_count = self.R
                            start_pos = center_pos - target_count // 2  # 左边起始位置
                            end_pos = center_pos + target_count // 2    # 右边结束位置
                            
                            # 确保选择范围在有效范围内 (0 到 N-1)
                            start_pos = max(0, start_pos)
                            end_pos = min(N, end_pos)
                            
                            # 生成连续的token索引，确保空间连续性
                            visual_token_index = torch.arange(start_pos, end_pos, device=image_attention.device)
                            
                        else:
                            # 方法2: 基于attention值选择 - 选择attention值最高的self.R个token
                            # 使用新的聚合后的attention分布
                            image_attention_selected = final_image_attention  # (N)
                            
                            # 使用top-k选择attention值最高的self.R个token，然后排序
                            visual_token_length = self.R
                            visual_token_index = torch.topk(image_attention_selected, k=self.R).indices # (R)
                            visual_token_index = torch.sort(visual_token_index).values # (R)
                        
                        # 保存第K层的注意力分数
                        layer_attention_scores.append({
                            'layer': decoder_layer.self_attn.layer_idx + 1,
                            'attention_scores': image_attention.detach().cpu().numpy(),
                            'head_attention': head_attention.detach().cpu().numpy(),
                            'visual_head_index': visual_head_index.detach().cpu().numpy()
                        })

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
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_cache
        
        # 保存所有层的注意力分数
        if seq_length > 1:
            self.layer_attention_scores = layer_attention_scores
            # 直接保存all_self_attns，包含所有层的完整attention信息
            torch.save(all_self_attns, "all_layers_attention_scores.pt")
            print("All layers attention scores saved to: all_layers_attention_scores.pt")
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
