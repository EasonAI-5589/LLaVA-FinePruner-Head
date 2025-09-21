#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .modeling_llama_fastv import FastVLlamaModel
from .modeling_llama_fastv_fine import FineFastVLlamaModel
from .modeling_llama_fastv_ablation_a import FineFastVLlamaModelAblationA
from .modeling_llama_fastv_ablation_dynamic import FineFastVLlamaModelAblationDynamic
from .modeling_llama_pdrop import PDropLlamaModel
from .modeling_llama_sparsevlm import SparseLlamaModel
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class FastVLlavaLlamaModel(LlavaMetaModel, FastVLlamaModel):
    # Alter LlavaLlamaModel to FastVLlavaLlamaModel
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super(FastVLlavaLlamaModel, self).__init__(config, visual_token_num=visual_token_num)


class FineFastVLlavaLlamaModel(LlavaMetaModel, FineFastVLlamaModel):
    # Alter LlavaLlamaModel to FineFastVLlavaLlamaModel
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super(FineFastVLlavaLlamaModel, self).__init__(config, visual_token_num=visual_token_num)


class FineFastVLlavaLlamaModelAblationA(LlavaMetaModel, FineFastVLlamaModelAblationA):
    # 消融研究A: 头筛选策略对比
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super(FineFastVLlavaLlamaModelAblationA, self).__init__(config, visual_token_num=visual_token_num)


class FineFastVLlavaLlamaModelAblationDynamic(LlavaMetaModel, FineFastVLlamaModelAblationDynamic):
    # 动态Head选择: 层次化智能选择
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super(FineFastVLlavaLlamaModelAblationDynamic, self).__init__(config, visual_token_num=visual_token_num)


class PDropLlavaLlamaModel(LlavaMetaModel, PDropLlamaModel):
    # Alter LlavaLlamaModel to PDropLlavaLlamaModel
    config_class = LlavaLlamaConfig
    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super(PDropLlavaLlamaModel, self).__init__(config, visual_token_num=visual_token_num)


class SparseLlavaLlamaModel(LlavaMetaModel, SparseLlamaModel):
    # Alter LlavaLlamaModel to SparseLlavaLlamaModel
    config_class = LlavaLlamaConfig
    def __init__(self, config: LlamaConfig, visual_token_num: int):
        super(SparseLlavaLlamaModel, self).__init__(config, visual_token_num=visual_token_num)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config, pruning_method=None, visual_token_num=None, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        if pruning_method == "fastv":
            print(f"Use FastV!")
            self.model = FastVLlavaLlamaModel(config, visual_token_num)
        elif pruning_method == "fastv+finepruner":
            print(f"Use FineFastV!")
            self.model = FineFastVLlavaLlamaModel(config, visual_token_num)
        elif pruning_method == "ablation_a":
            print(f"Use Ablation A (Head Selection)!")
            self.model = FineFastVLlavaLlamaModelAblationA(config, visual_token_num)
        elif pruning_method == "dynamic_head":
            print(f"Use Dynamic Head Selection!")
            self.model = FineFastVLlavaLlamaModelAblationDynamic(config, visual_token_num)
        elif pruning_method == "pdrop":
            print(f"Use PDrop!")
            self.model = PDropLlavaLlamaModel(config, visual_token_num)
        elif pruning_method == "sparsevlm":
            print(f"Use SparseVLM!")
            self.model = SparseLlavaLlamaModel(config, visual_token_num)
        else:
            self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                visual_token_num
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        ), visual_token_num

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
