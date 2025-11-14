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

    def set_lora_second_moment_map(self, second_moment_map: dict[str, torch.Tensor], adapter_name: str):
        # Now inject second moment into all relevant modules
        for name, module in self.model.named_modules():
            if hasattr(module, "set_lora_second_moment") and name in second_moment_map:
                module.set_lora_second_moment(second_moment=second_moment_map[name], adapter_name=adapter_name)

    def set_lora_S_KpKp_map(self, S_KpKp_map: dict[str, torch.Tensor], S_count_map: dict[str, int], adapter_name: str):
        # Now inject S_KpKp into all relevant modules
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_S_KpKp"):
                module.set_lora_S_KpKp(S_KpKp=S_KpKp_map.get(name, None), S_count=S_count_map.get(name, 1), adapter_name=adapter_name)

    def set_lora_S_KpVp_map(self, S_KpVp_map: dict[str, torch.Tensor], S_count_map: dict[str, int], adapter_name: str):
        # Now inject S_KpVp into all relevant modules
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_S_KpVp"):
                module.set_lora_S_KpVp(S_KpVp=S_KpVp_map.get(name, None), S_count=S_count_map.get(name, 1), adapter_name=adapter_name)

    def set_lora_S_VpVp_map(self, S_VpVp_map: dict[str, torch.Tensor], S_count_map: dict[str, int], adapter_name: str):
        # Now inject S_VpVp into all relevant modules
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_S_VpVp"):
                module.set_lora_S_VpVp(S_VpVp=S_VpVp_map.get(name, None), S_count=S_count_map.get(name, 1), adapter_name=adapter_name)

    def get_delta_weights(self, adapter: str) -> dict[str, torch.Tensor]:
        delta_weights = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                delta_weights[name] = module.get_delta_weight(adapter=adapter)
        return delta_weights

    def get_delta_KpKp(self, adapter: str) -> dict[str, torch.Tensor]:
        delta_KpKp = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                delta_KpKp[name] = module.get_delta_KpKp(adapter=adapter)
        return delta_KpKp

    def get_previous_losses_with_trace(self, adapter: str) -> dict[str, torch.Tensor]:
        trace_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                trace_dict[name] = module.get_previous_loss_with_trace(adapter=adapter)
        return trace_dict

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
