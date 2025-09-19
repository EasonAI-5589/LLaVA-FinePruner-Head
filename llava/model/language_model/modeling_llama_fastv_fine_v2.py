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


class FineFastVLlamaModelV2(LlamaModel):
    """
    基于text-visual token注意力交互的改进版本
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
        self.head_selection_strategy = getattr(config, 'head_selection_strategy', 'text_visual_global')
        # 'text_visual_global', 'text_to_visual', 'visual_importance_accumulation', 'cross_modal_interaction'

    def select_heads_text_visual_global(self, last_attention):
        """方法1：基于text-visual全局注意力交互选择头"""
        # 获取text token对visual token的注意力
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 如果没有text token，回退到原始方法
            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
            head_attention = image_attention.sum(dim=-1)
            visual_head_index = head_attention.topk(k=self.H).indices
            return visual_head_index, image_attention[visual_head_index].mean(dim=0)

        # text token对visual token的注意力 (H, text_len, visual_len)
        text_to_visual_attention = last_attention[0, :, text_start:text_end,
                                                 self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 计算每个头的text-visual交互强度
        head_interaction_strength = text_to_visual_attention.sum(dim=(1, 2))  # (H,) - 对text和visual维度求和
        visual_head_index = head_interaction_strength.topk(k=self.H).indices

        # 选中头的平均注意力模式
        selected_attention = text_to_visual_attention[visual_head_index].mean(dim=(0, 1))  # (visual_len,)

        return visual_head_index, selected_attention

    def select_heads_text_to_visual(self, last_attention):
        """方法2：基于text token对visual token的注意力强度"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 回退方案
            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
            head_attention = image_attention.sum(dim=-1)
            visual_head_index = head_attention.topk(k=self.H).indices
            return visual_head_index, image_attention[visual_head_index].mean(dim=0)

        text_to_visual_attention = last_attention[0, :, text_start:text_end,
                                                 self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 方法2a: 选择对visual token关注度高的头
        avg_attention_per_head = text_to_visual_attention.mean(dim=1).sum(dim=1)  # (H,) - 每个头对visual token的平均关注度
        visual_head_index = avg_attention_per_head.topk(k=self.H).indices

        # 计算visual token重要性：被text token关注得越多越重要
        visual_importance = text_to_visual_attention[visual_head_index].mean(dim=(0, 1))  # (visual_len,)

        return visual_head_index, visual_importance

    def select_heads_visual_importance_accumulation(self, last_attention):
        """方法3：基于visual token重要性累积"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 回退方案
            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
            head_attention = image_attention.sum(dim=-1)
            visual_head_index = head_attention.topk(k=self.H).indices
            return visual_head_index, image_attention[visual_head_index].mean(dim=0)

        text_to_visual_attention = last_attention[0, :, text_start:text_end,
                                                 self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 每个头中，每个visual token被所有text token关注的总和
        visual_token_importance_per_head = text_to_visual_attention.sum(dim=1)  # (H, visual_len)

        # 选择能识别出重要visual token的头（方差大的头）
        head_discrimination = visual_token_importance_per_head.var(dim=1)  # (H,)
        visual_head_index = head_discrimination.topk(k=self.H).indices

        # 累积选中头的visual token重要性
        accumulated_importance = visual_token_importance_per_head[visual_head_index].mean(dim=0)  # (visual_len,)

        return visual_head_index, accumulated_importance

    def select_heads_cross_modal_interaction(self, last_attention):
        """方法4：基于跨模态交互模式"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 回退方案
            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
            head_attention = image_attention.sum(dim=-1)
            visual_head_index = head_attention.topk(k=self.H).indices
            return visual_head_index, image_attention[visual_head_index].mean(dim=0)

        # text-to-visual attention
        text_to_visual = last_attention[0, :, text_start:text_end,
                                       self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # visual-to-text attention (如果有的话)
        visual_to_text = last_attention[0, :, self.system_prompt_length:self.system_prompt_length+self.visual_token_length,
                                       text_start:text_end]

        # 计算双向交互强度
        text_to_visual_strength = text_to_visual.sum(dim=(1, 2))  # (H,)
        visual_to_text_strength = visual_to_text.sum(dim=(1, 2))  # (H,)

        # 综合交互强度
        cross_modal_strength = text_to_visual_strength + visual_to_text_strength
        visual_head_index = cross_modal_strength.topk(k=self.H).indices

        # 结合双向信息计算visual token重要性
        forward_importance = text_to_visual[visual_head_index].mean(dim=(0, 1))  # text->visual
        backward_importance = visual_to_text[visual_head_index].mean(dim=(0, 2))  # visual->text 的反向重要性

        # 加权结合
        combined_importance = 0.7 * forward_importance + 0.3 * backward_importance

        return visual_head_index, combined_importance

    def select_heads_adaptive_text_visual(self, last_attention):
        """方法5：自适应text-visual注意力（考虑文本长度）"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 没有text token时的处理
            image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
            head_attention = image_attention.sum(dim=-1)
            visual_head_index = head_attention.topk(k=self.H).indices
            return visual_head_index, image_attention[visual_head_index].mean(dim=0)

        text_len = text_end - text_start
        text_to_visual = last_attention[0, :, text_start:text_end,
                                       self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 根据文本长度调整策略
        if text_len <= 5:
            # 短文本：重点关注最后几个token
            recent_text_attention = text_to_visual[:, -min(text_len, 3):, :]  # 最后3个或更少
            head_scores = recent_text_attention.sum(dim=(1, 2))
        else:
            # 长文本：考虑全局模式，但给近期token更高权重
            weights = torch.linspace(0.5, 1.0, text_len, device=text_to_visual.device)  # 线性递增权重
            weighted_attention = text_to_visual * weights.view(1, -1, 1)
            head_scores = weighted_attention.sum(dim=(1, 2))

        visual_head_index = head_scores.topk(k=self.H).indices

        # 计算加权的visual token重要性
        if text_len <= 5:
            visual_importance = recent_text_attention[visual_head_index].mean(dim=(0, 1))
        else:
            weighted_selected = (text_to_visual * weights.view(1, -1, 1))[visual_head_index]
            visual_importance = weighted_selected.mean(dim=(0, 1))

        return visual_head_index, visual_importance

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

                # FastV with text-visual attention based head selection
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

                        # 根据策略选择注意力头和计算visual token重要性
                        if self.head_selection_strategy == 'text_visual_global':
                            visual_head_index, image_attention = self.select_heads_text_visual_global(last_attention)
                        elif self.head_selection_strategy == 'text_to_visual':
                            visual_head_index, image_attention = self.select_heads_text_to_visual(last_attention)
                        elif self.head_selection_strategy == 'visual_importance_accumulation':
                            visual_head_index, image_attention = self.select_heads_visual_importance_accumulation(last_attention)
                        elif self.head_selection_strategy == 'cross_modal_interaction':
                            visual_head_index, image_attention = self.select_heads_cross_modal_interaction(last_attention)
                        elif self.head_selection_strategy == 'adaptive_text_visual':
                            visual_head_index, image_attention = self.select_heads_adaptive_text_visual(last_attention)
                        else:
                            # 默认：text_visual_global
                            visual_head_index, image_attention = self.select_heads_text_visual_global(last_attention)

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