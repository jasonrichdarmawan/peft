import torch
import torch.nn as nn
from typing import Union

from transformers.pytorch_utils import Conv1D

from peft.utils.other import transpose, BufferDict
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
        self.lora_P = BufferDict()
        # self.lora_S = BufferDict()
        # self.lora_S_pending = BufferDict()

    def set_lora_P(self, lora_P: torch.Tensor, adapter_name: str):
        if lora_P.shape[0] != self.in_features or lora_P.shape[1] != self.in_features:
            raise ValueError(f"P matrix shape {self.P.shape} does not match in_features {self.in_features}.")
        self.lora_P[adapter_name] = lora_P

    # def update_lora_S(self, lora_S: torch.Tensor, adapter_name: str):
    #     if adapter_name not in self.lora_S_pending:
    #         if lora_S.shape[0] != self.in_features or lora_S.shape[1] != self.in_features:
    #             raise ValueError(f"S matrix shape {lora_S.shape} does not match in_features {self.in_features}.")
    #         self.lora_S_pending[adapter_name] = lora_S.clone().detach()
    #     else:
    #         self.lora_S_pending[adapter_name] += lora_S
    
    # def merge_lora_S(self, adapter_name: str):
    #     if adapter_name not in self.lora_S_pending:
    #         raise ValueError(f"No pending S matrix to merge for adapter {adapter_name}.")
    #     if adapter_name not in self.lora_S:
    #         self.lora_S[adapter_name] = self.lora_S_pending[adapter_name]
    #     else:
    #         self.lora_S[adapter_name] += self.lora_S_pending[adapter_name]
    #     del self.lora_S_pending[adapter_name]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # lora_K_p_attention_mask = kwargs.pop("lora_K_p_attention_mask", None)

        if self.disable_adapters:
            if self.merged:
                self._unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
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
                lora_P = self.lora_P[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                # if lora_K_p_attention_mask is not None:
                #     with torch.no_grad():
                #         flat_x = x.flatten(start_dim=0, end_dim=1)  # (B*S, F)
                #         attended_tokens = lora_K_p_attention_mask.flatten().nonzero()[:, 0].to(flat_x.device)
                #         flat_x = flat_x[attended_tokens, :]
                #         K_pK_p_new = flat_x.T @ flat_x
                #         self.update_lora_S(lora_S=K_pK_p_new, adapter_name=active_adapter)

                if not self.use_dora[active_adapter]:
                    # P is a projection matrix, so P.T == P
                    # U \Lambda U^T = SVD(K_0 K_0^T)
                    # P = UU^T
                    result = result + (lora_B(lora_A(dropout(x) @ lora_P))) * scaling
                else:
                    raise NotImplementedError("DoRa is not implemented yet.")

            result = result.to(torch_result_dtype)

        return result

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        P = self.lora_P[adapter]

        output_tensor = transpose(weight_B @ weight_A @ P, fan_in_fan_out=self.fan_in_fan_out) * self.scaling[adapter]

        return output_tensor

    # def get_delta_weight_K_p(self, adapter: str) -> torch.Tensor:
    #     delta_weight = self.get_delta_weight(adapter)
    #     lora_S = self.lora_S[adapter]
    #     if lora_S is None:
    #         return torch.zeros((delta_weight.shape[0], delta_weight.shape[0]), device=delta_weight.device, dtype=delta_weight.dtype)
    #     delta_K_p = delta_weight @ lora_S @ delta_weight.T
    #     return delta_K_p


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
        new_module = Embedding(base_layer=target, adapter_name=adapter_name, **embedding_kwargs)
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
        new_module = NullSpaceLinear(base_layer=target, adapter_name=adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
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
