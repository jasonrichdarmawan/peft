import torch
import torch.nn as nn

from peft.tuners.lora import LoraModel, LoraLayer, LoraConfig
from peft.config import PeftConfig

from transformers import PreTrainedModel

from itertools import chain
import re

from .layer import dispatcher_default


class NullSpaceLoraModel(LoraModel):
    def __init__(
        self,
        model: PreTrainedModel,
        config: PeftConfig,
        adapter_name: str,
    ):
        super().__init__(model=model, config=config, adapter_name=adapter_name)

    def set_lora_P_map(self, lora_P_map: dict[str, torch.Tensor], adapter_name: str):
        # Now inject P into all relevant modules
        for name, module in self.model.named_modules():
            if hasattr(module, "set_lora_P") and name in lora_P_map:
                module.set_lora_P(lora_P=lora_P_map[name], adapter_name=adapter_name)

    def get_delta_weights(self, adapter: str) -> dict[str, torch.Tensor]:
        delta_weights = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                delta_weights[name] = module.get_delta_weight(adapter)
        return delta_weights

    @staticmethod
    def _create_new_module(lora_config: LoraConfig, adapter_name: str, target: nn.Module, **kwargs):
        dispatchers = []

        # dispatch_bnb_8bit, dispatch_bnb_4bit
        # dispatch_aqlm, dispatch_awq, dispatch_gptq,
        # dispatch_megatron not implemented yet

        dispatchers.extend([dispatcher_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module
