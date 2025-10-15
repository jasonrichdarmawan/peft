import torch
import torch.nn as nn
from typing import Union

from transformers.pytorch_utils import Conv1D

from peft.utils.other import transpose
from peft.tuners.lora import Linear, Embedding, Conv2d, LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer

import warnings

class NullSpaceLinear(Linear):

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace store weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        super().__init__(
            base_layer=base_layer,
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            is_target_conv_1d_layer=is_target_conv_1d_layer,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            **kwargs,
        )

    def set_P(self, P: torch.Tensor):
        self.P = P
        if self.P.shape[0] != self.in_features or self.P.shape[1] != self.in_features:
            raise ValueError(
                f"P matrix shape {self.P.shape} does not match in_features {self.in_features}."
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self._unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs
            )
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # P is a projection matrix, so P.T == P
                    # U \Lambda U^T = SVD(K_0 K_0^T)
                    # P = UU^T
                    result = result + (lora_B(lora_A(dropout(x) @ self.P))) * scaling
                else:
                    raise NotImplementedError("DoRa is not implemented yet.")

            result = result.to(torch_result_dtype)

        return result

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        output_tensor = (
            transpose(weight_B @ weight_A @ self.P, fan_in_fan_out=self.fan_in_fan_out)
            * self.scaling[adapter]
        )

        return output_tensor

def dispatcher_default(
    target: nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(
            base_layer=target, adapter_name=adapter_name, **embedding_kwargs
        )
    elif isinstance(target_base_layer, nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(base_layer=target, adapter_name=adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = NullSpaceLinear(
            base_layer=target, adapter_name=adapter_name, **kwargs
        )
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = NullSpaceLinear(
            base_layer=target,
            adapter_name=adapter_name,
            is_target_conv_1d_layer=True,
            **kwargs,
        )

    return new_module