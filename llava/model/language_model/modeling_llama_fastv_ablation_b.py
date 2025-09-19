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


class FineFastVLlamaModelAblationB(LlamaModel):
    """
    消融研究B: Query Token选择策略比较
    固定使用某种头筛选方法，比较不同的query token选择策略效果
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

        # 消融研究B: query token选择策略
        self.query_selection_strategy = getattr(config, 'query_selection_strategy', 'last_token')
        # 固定使用的头筛选方法 (建议使用表现较好的方法，如'variance')
        self.fixed_head_selection = getattr(config, 'fixed_head_selection', 'variance')

    def fixed_head_selection_method(self, image_attention):
        """固定的头筛选方法 - 使用方差方法作为默认"""
        if self.fixed_head_selection == 'sum':
            head_scores = image_attention.sum(dim=-1)
        elif self.fixed_head_selection == 'variance':
            head_scores = image_attention.var(dim=-1)
        elif self.fixed_head_selection == 'entropy':
            attention_probs = torch.softmax(image_attention, dim=-1)
            head_scores = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1)
            return head_scores.topk(k=self.H, largest=False).indices
        else:
            # 默认使用方差
            head_scores = image_attention.var(dim=-1)

        return head_scores.topk(k=self.H).indices

    def query_strategy_last_token(self, last_attention):
        """策略1: 使用最后一个token (Baseline)"""
        image_attention = last_attention[0, :, -1, self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
        visual_head_index = self.fixed_head_selection_method(image_attention)
        final_attention = image_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_last_n_tokens(self, last_attention, n=3):
        """策略2: 使用最后N个token的平均"""
        seq_len = last_attention.size(2)
        start_idx = max(0, seq_len - n)

        # 获取最后N个token的注意力
        multi_token_attention = last_attention[0, :, start_idx:seq_len,
                                             self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 对query维度求平均
        avg_attention = multi_token_attention.mean(dim=1)  # (H, N)

        visual_head_index = self.fixed_head_selection_method(avg_attention)
        final_attention = avg_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_text_tokens_only(self, last_attention):
        """策略3: 仅使用text token (如果有的话)"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 没有text token，回退到最后一个token
            return self.query_strategy_last_token(last_attention)

        # 使用所有text token对visual token的注意力
        text_to_visual = last_attention[0, :, text_start:text_end,
                                       self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 对text token维度求平均
        avg_text_attention = text_to_visual.mean(dim=1)  # (H, N)

        visual_head_index = self.fixed_head_selection_method(avg_text_attention)
        final_attention = avg_text_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_weighted_recent(self, last_attention, window_size=5):
        """策略4: 对最近的token使用递增权重"""
        seq_len = last_attention.size(2)
        start_idx = max(0, seq_len - window_size)
        actual_window = seq_len - start_idx

        # 获取最近window_size个token的注意力
        recent_attention = last_attention[0, :, start_idx:seq_len,
                                        self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 创建递增权重 (更近的token权重更高)
        weights = torch.linspace(0.5, 1.0, actual_window, device=recent_attention.device)
        weights = weights / weights.sum()  # 归一化

        # 加权平均
        weighted_attention = (recent_attention * weights.view(1, -1, 1)).sum(dim=1)  # (H, N)

        visual_head_index = self.fixed_head_selection_method(weighted_attention)
        final_attention = weighted_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_system_and_text(self, last_attention):
        """策略5: 使用system prompt后的所有token (排除visual token)"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        if text_end <= text_start:
            # 没有text token，使用system prompt的最后几个token
            system_end = self.system_prompt_length
            system_start = max(0, system_end - 3)
            query_attention = last_attention[0, :, system_start:system_end,
                                           self.system_prompt_length:self.system_prompt_length+self.visual_token_length]
        else:
            # 使用所有非visual token
            all_query_indices = torch.cat([
                torch.arange(self.system_prompt_length, device=last_attention.device),
                torch.arange(text_start, text_end, device=last_attention.device)
            ])
            query_attention = last_attention[0, :, all_query_indices,
                                           self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 对query维度求平均
        avg_attention = query_attention.mean(dim=1)  # (H, N)

        visual_head_index = self.fixed_head_selection_method(avg_attention)
        final_attention = avg_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_attention_weighted(self, last_attention):
        """策略6: 基于注意力强度的加权组合"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        # 使用非visual token
        if text_end > text_start:
            # 有text token时，使用所有非visual token
            query_indices = torch.cat([
                torch.arange(self.system_prompt_length, device=last_attention.device),
                torch.arange(text_start, text_end, device=last_attention.device)
            ])
        else:
            # 没有text token时，使用system prompt
            query_indices = torch.arange(self.system_prompt_length, device=last_attention.device)

        query_attention = last_attention[0, :, query_indices,
                                       self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 计算每个query token对visual token的总注意力作为权重
        query_weights = query_attention.sum(dim=(0, 2))  # (query_len,)
        query_weights = torch.softmax(query_weights, dim=0)  # 归一化

        # 加权平均
        weighted_attention = (query_attention * query_weights.view(1, -1, 1)).sum(dim=1)  # (H, N)

        visual_head_index = self.fixed_head_selection_method(weighted_attention)
        final_attention = weighted_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_max_attention_token(self, last_attention):
        """策略7: 选择对visual token注意力最强的单个query token"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)

        # 使用非visual token
        if text_end > text_start:
            query_indices = torch.cat([
                torch.arange(self.system_prompt_length, device=last_attention.device),
                torch.arange(text_start, text_end, device=last_attention.device)
            ])
        else:
            query_indices = torch.arange(self.system_prompt_length, device=last_attention.device)

        query_attention = last_attention[0, :, query_indices,
                                       self.system_prompt_length:self.system_prompt_length+self.visual_token_length]

        # 找到对visual token注意力总和最大的query token
        query_strength = query_attention.sum(dim=(0, 2))  # (query_len,)
        best_query_idx = query_strength.argmax()

        # 使用最佳query token的注意力
        best_attention = query_attention[:, best_query_idx, :]  # (H, N)

        visual_head_index = self.fixed_head_selection_method(best_attention)
        final_attention = best_attention[visual_head_index].mean(dim=0)
        return final_attention

    def query_strategy_adaptive_by_length(self, last_attention):
        """策略8: 根据序列长度自适应选择策略"""
        text_start = self.system_prompt_length + self.visual_token_length
        text_end = last_attention.size(2)
        text_length = text_end - text_start

        if text_length == 0:
            # 没有text token，使用最后一个token
            return self.query_strategy_last_token(last_attention)
        elif text_length <= 3:
            # 短text，使用所有text token
            return self.query_strategy_text_tokens_only(last_attention)
        elif text_length <= 10:
            # 中等长度，使用加权最近token
            return self.query_strategy_weighted_recent(last_attention, window_size=min(5, text_length))
        else:
            # 长text，使用基于注意力强度的加权
            return self.query_strategy_attention_weighted(last_attention)

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

                # 消融研究B: query token选择策略比较
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

                        # 根据策略选择query token和计算visual token重要性
                        if self.query_selection_strategy == 'last_token':
                            image_attention = self.query_strategy_last_token(last_attention)
                        elif self.query_selection_strategy == 'last_n_tokens':
                            image_attention = self.query_strategy_last_n_tokens(last_attention, n=3)
                        elif self.query_selection_strategy == 'last_5_tokens':
                            image_attention = self.query_strategy_last_n_tokens(last_attention, n=5)
                        elif self.query_selection_strategy == 'text_tokens_only':
                            image_attention = self.query_strategy_text_tokens_only(last_attention)
                        elif self.query_selection_strategy == 'weighted_recent':
                            image_attention = self.query_strategy_weighted_recent(last_attention)
                        elif self.query_selection_strategy == 'system_and_text':
                            image_attention = self.query_strategy_system_and_text(last_attention)
                        elif self.query_selection_strategy == 'attention_weighted':
                            image_attention = self.query_strategy_attention_weighted(last_attention)
                        elif self.query_selection_strategy == 'max_attention_token':
                            image_attention = self.query_strategy_max_attention_token(last_attention)
                        elif self.query_selection_strategy == 'adaptive_by_length':
                            image_attention = self.query_strategy_adaptive_by_length(last_attention)
                        else:
                            # 默认使用最后一个token
                            image_attention = self.query_strategy_last_token(last_attention)

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