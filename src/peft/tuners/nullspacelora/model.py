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

    def set_P_map(self, P_map: dict[str, torch.Tensor]):
        self.P_map = P_map
        # Now inject P into all relevant modules
        for name, module in self.model.named_modules():
            if hasattr(module, "set_P") and name in P_map:
                module.set_P(P_map[name])

    def _create_and_replace(
        self,
        lora_config: LoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(
            filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys),
            current_key,
        )
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        # quant_method is not implemented yet

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

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
