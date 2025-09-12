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
    
                        # FastV implementation - 根据情况选择不同的attention计算方式
                        use_last_token = False   # 设置为True使用最后一个token方法
                        use_text_token = True  # 设置为True使用text+generated token方法
                        
                        if use_last_token:
                            # 方法1: 使用最后一个token对visual token的attention (原始FastV方法)
                            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+visual_token_length] # (H, N)
                        elif use_text_token:
                            # 方法2: 取所有text token和generated token对visual token的attention
                            # last_attention shape: (batch_size, num_heads, seq_length, seq_length)
                            # 我们取从visual token之后的所有token对visual token的attention
                            text_and_generated_tokens = last_attention[0, :, self.system_prompt_length+visual_token_length:, self.system_prompt_length:self.system_prompt_length+visual_token_length] # (H, T_text, N)
                            
                            # 复杂筛选方案：多阶段注意力优化
                            # 阶段1: 计算text token的重要性分数
                            text_importance = text_and_generated_tokens.sum(dim=-1) # (H, T_text)
                            
                            # 阶段2: 动态top-k筛选重要的text token
                            # 方案A: 基于重要性阈值的动态筛选
                            importance_threshold = text_importance.mean() + 0.5 * text_importance.std()  # 动态阈值
                            # 计算超过阈值的text token数量
                            above_threshold = (text_importance > importance_threshold.unsqueeze(-1)).sum(dim=-1)  # (H,)
                            # 确保至少保留3个，最多保留min(12, total_text_tokens)
                            k_text = torch.clamp(above_threshold, min=3, max=min(12, text_and_generated_tokens.shape[1]))
                            
                            # 方案B: 基于重要性分布的百分位数筛选（注释掉，可选择使用）
                            # # 使用75%分位数作为阈值
                            # percentile_75 = torch.quantile(text_importance, 0.75, dim=-1)  # (H,)
                            # above_percentile = (text_importance > percentile_75.unsqueeze(-1)).sum(dim=-1)
                            # k_text = torch.clamp(above_percentile, min=2, max=min(15, text_and_generated_tokens.shape[1]))
                            
                            # 方案C: 基于重要性衰减的自适应筛选（注释掉，可选择使用）
                            # # 计算重要性排序后的衰减率
                            # sorted_importance, _ = text_importance.sort(dim=-1, descending=True)
                            # decay_ratio = sorted_importance[:, 1:] / (sorted_importance[:, :-1] + 1e-8)
                            # # 找到衰减率超过阈值的第一个位置
                            # decay_threshold = 0.7  # 衰减率阈值
                            # significant_positions = (decay_ratio > decay_threshold).sum(dim=-1) + 1
                            # k_text = torch.clamp(significant_positions, min=3, max=min(10, text_and_generated_tokens.shape[1]))
                            
                            # 执行动态top-k筛选
                            top_k_text_importance, top_k_indices = text_importance.topk(k_text.max().item(), dim=-1) # (H, max_k)
                            # 根据每个head的实际k_text值进行masking
                            mask = torch.arange(top_k_text_importance.shape[1], device=top_k_text_importance.device).unsqueeze(0) < k_text.unsqueeze(1)
                            top_k_text_importance = top_k_text_importance * mask.float()
                            top_k_indices = top_k_indices * mask.long()
                            
                            # 阶段3: 基于top-k text token重新计算attention
                            top_k_attention = text_and_generated_tokens.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, text_and_generated_tokens.shape[-1])) # (H, k_text, N)
                            
                            # 阶段4: 计算visual token的敏感度分数
                            visual_sensitivity = top_k_attention.sum(dim=1) # (H, N)
                            
                            # 阶段5: 使用sigmoid激活突出敏感的visual token
                            sensitivity_weights = torch.sigmoid(visual_sensitivity / visual_sensitivity.std()) # (H, N)
                            
                            # 阶段6: 最终聚合 - 结合text重要性和visual敏感度
                            text_weights = torch.softmax(top_k_text_importance, dim=-1) # (H, k_text)
                            weighted_attention = (top_k_attention * text_weights.unsqueeze(-1)) # (H, k_text, N)
                            
                            # 阶段7: 应用visual敏感度权重，得到最终的image_attention
                            image_attention = (weighted_attention.sum(dim=1)) * sensitivity_weights # (H, N)
                        else:
                            # 默认使用最后一个token方法
                            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+visual_token_length] # (H, N)


                        head_attention = image_attention.sum(dim=-1)
                        visual_head_index = head_attention.topk(k=self.H).indices # (H)
                        
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
                            # 计算选中head的平均attention分布
                            image_attention_selected = image_attention[visual_head_index].mean(dim=0)  # (N)
                            
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
