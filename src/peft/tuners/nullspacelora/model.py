import torch
import torch.nn as nn

from peft.tuners.lora import LoraModel, LoraLayer, LoraConfig
from peft.tuners.lora.model import _adapter_names_pre_forward_hook
from peft.config import PeftConfig

from transformers import PreTrainedModel

from contextlib import contextmanager
from functools import partial

from .layer import dispatcher_default


# def _lora_K_p_attention_mask_pre_forward_hook(module, args, kwargs, lora_K_p_attention_mask: torch.Tensor):
#     kwargs["lora_K_p_attention_mask"] = lora_K_p_attention_mask
#     return args, kwargs


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
                delta_weights[name] = module.get_delta_weight(adapter=adapter)
        return delta_weights

    # def merge_lora_S(self, adapter: str):
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, LoraLayer):
    #             module.merge_lora_S(adapter_name=adapter)

    # def get_delta_weights_K_p(self, adapter: str) -> dict[str, torch.Tensor]:
    #     delta_weights_K_p = {}
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, LoraLayer):
    #             delta_weights_K_p[name] = module.get_delta_weight_K_p(adapter=adapter)
    #     return delta_weights_K_p

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

    # @contextmanager
    # def _enable_peft_forward_hooks(self, *args, **kwargs):
    #     # If adapter_names is passed as an argument, we inject it into the forward arguments.
    #     adapter_names = kwargs.pop("adapter_names", None)

    #     lora_K_p_attention_mask = kwargs.pop("lora_K_p_attention_mask", None)

    #     if adapter_names is None and lora_K_p_attention_mask is None:
    #         # nothing to do
    #         yield
    #         return

    #     if adapter_names is not None and self.training:
    #         raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

    #     hook_handles = []
    #     for module in self.modules():
    #         if isinstance(module, LoraLayer):
    #             if adapter_names is not None:
    #                 pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
    #                 handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
    #                 hook_handles.append(handle)
    #             if lora_K_p_attention_mask is not None:
    #                 pre_forward = partial(
    #                     _lora_K_p_attention_mask_pre_forward_hook, lora_K_p_attention_mask=lora_K_p_attention_mask
    #                 )
    #                 handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
    #                 hook_handles.append(handle)

    #     yield

    #     for handle in hook_handles:
    #         handle.remove()
