import torch
import torch.nn as nn
from typing import Union, Any, Optional

from transformers.pytorch_utils import Conv1D

from peft.utils.other import transpose, BufferDict
from peft.tuners.lora import Linear, Embedding, Conv2d, LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

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
        nullspace_threshold: float = 2e-2,
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
        self.lora_second_moment = BufferDict()
        self.nullspace_threshold = nullspace_threshold
        self.lora_P_2 = BufferDict()
        self.lora_S_KpKp = BufferDict()
        self.lora_S_KpVp = BufferDict()
        self.lora_S_VpVp = BufferDict()
        self.lora_S_count = BufferDict()

    def set_lora_second_moment(self, second_moment: dict[str, Any], adapter_name: str):
        self.lora_second_moment[f"{adapter_name}_mom2"] = second_moment["mom2"]
        self.lora_second_moment[f"{adapter_name}_count"] = torch.tensor(
            second_moment["count"], dtype=torch.long, device=second_moment["mom2"].device
        )
        self.set_lora_P_2_from_second_moment(adapter_name=adapter_name)

    def set_lora_P_2_from_second_moment(self, adapter_name: str):
        mom2 = self.lora_second_moment[f"{adapter_name}_mom2"].to(self.lora_A[adapter_name].weight.device)
        count = self.lora_second_moment[f"{adapter_name}_count"].to(self.lora_A[adapter_name].weight.device)
        moment = mom2 / count
        eigenvalues, eigenvectors = torch.linalg.eigh(moment)
        small_eigenvalues_indices = (eigenvalues < self.nullspace_threshold).nonzero(as_tuple=True)[0]
        P_2 = eigenvectors[:, small_eigenvalues_indices]
        self._set_lora_P_2(P_2, adapter_name)

    def _set_lora_P_2(self, lora_P_2: torch.Tensor, adapter_name: str):
        # use output statistics
        # if lora_P_2.shape[0] != self.out_features:
        #     raise ValueError(f"P_2 matrix shape {self.lora_P_2.shape} does not match out_features {self.out_features}.")
        
        # use input statistics
        if lora_P_2.shape[0] != self.in_features:
            raise ValueError(f"P_2 matrix shape {lora_P_2.shape} does not match in_features {self.in_features}.")
        
        self.lora_P_2[adapter_name] = lora_P_2

    def set_lora_S_KpKp(self, S_KpKp: torch.Tensor, adapter_name: str):
        if S_KpKp == None:
            self.lora_S_KpKp[adapter_name] = torch.zeros(
                self.in_features, self.in_features, device=self.lora_A[adapter_name].weight.device
            )
        else:
            if S_KpKp.shape[0] != self.in_features or S_KpKp.shape[1] != self.in_features:
                raise ValueError(f"S_KpKp matrix shape {S_KpKp.shape} does not match in_features {self.in_features}.")
            self.lora_S_KpKp[adapter_name] = S_KpKp.to(self.lora_A[adapter_name].weight.device)

    def set_lora_S_KpVp(self, S_KpVp: torch.Tensor, adapter_name: str):
        if S_KpVp == None:
            self.lora_S_KpVp[adapter_name] = torch.zeros(
                self.in_features, self.out_features, device=self.lora_A[adapter_name].weight.device
            )
        else:
            if S_KpVp.shape[0] != self.in_features or S_KpVp.shape[1] != self.out_features:
                raise ValueError(
                    f"S_KpVp matrix shape {S_KpVp.shape} does not match in_features {self.in_features} and out_features {self.out_features}."
                )
            self.lora_S_KpVp[adapter_name] = S_KpVp.to(self.lora_A[adapter_name].weight.device)

    def set_lora_S_VpVp(self, S_VpVp: torch.Tensor, adapter_name: str):
        if S_VpVp == None:
            self.lora_S_VpVp[adapter_name] = torch.zeros(
                self.out_features, self.out_features, device=self.lora_A[adapter_name].weight.device
            )
        else:
            if S_VpVp.shape[0] != self.out_features or S_VpVp.shape[1] != self.out_features:
                raise ValueError(
                    f"S_VpVp matrix shape {S_VpVp.shape} does not match out_features {self.out_features}."
                )
            self.lora_S_VpVp[adapter_name] = S_VpVp.to(self.lora_A[adapter_name].weight.device)

    def set_lora_S_count(self, S_count: int, adapter_name: str):
        if S_count is None:
            self.lora_S_count[adapter_name] = torch.tensor(1, dtype=torch.long, device=self.lora_A[adapter_name].weight.device)
        else:
            self.lora_S_count[adapter_name] = torch.tensor(S_count, dtype=torch.long, device=self.lora_A[adapter_name].weight.device)

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
                lora_P_2 = self.lora_P_2[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # U \Lambda U^T = SVD(K_0 K_0^T)
                    # P = UU^T
                    # So, P^T = P
                    
                    # use output statistics
                    # result = result + (lora_B(lora_A(dropout(x))) @ lora_P_2 @ lora_P_2.T) * scaling

                    # use input statistics
                    # x: (batch, seq, in)
                    # P_2: (in, k)
                    # x_proj = x @ P_2 @ P_2.T
                    x_projected = (x @ lora_P_2) @ lora_P_2.T  # Project input to null space
                    result = result + (lora_B(lora_A(dropout(x_projected)))) * scaling
                else:
                    raise NotImplementedError("DoRa is not implemented yet.")

            result = result.to(torch_result_dtype)

        return result

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        P_2 = self.lora_P_2[adapter]

        # use output statistics
        # # Low-rank order:
        # # step 1: (d, out) @ (out, r) = (d, r)
        # tmp = torch.matmul(P_2.T, weight_B)  # (d, r)
        # # step 2: (out, d) @ (d, r) = (out, r)
        # tmp = torch.matmul(P_2, tmp)  # (out, r)
        # # step 3: (out, r) @ (r, in) = (out, in)
        # out = torch.matmul(tmp, weight_A)  # (out, in)

        # output_tensor = transpose(out, fan_in_fan_out=self.fan_in_fan_out) * self.scaling[adapter]
        # return output_tensor

        # Alternative way (less efficient):
        # output_tensor = transpose(P_2 @ P_2.T @ weight_B @ weight_A, fan_in_fan_out=self.fan_in_fan_out) * self.scaling[adapter]

        # return output_tensor

        # use input statistics
        # Delta W = B @ A @ P @ P^T
        # B: (out, r)
        # A: (r, in)
        # P: (in, k)
        # 1. A @ P -> (r, k)
        AP = torch.matmul(weight_A, P_2)  # (r, k)
        # 2. B @ AP -> (out, k)
        BAP = torch.matmul(weight_B, AP)  # (out, k)
        # 3. BAP @ P^T -> (out, in)
        delta_W = torch.matmul(BAP, P_2.T)  # (out, in)
        output_tensor = transpose(delta_W, fan_in_fan_out=self.fan_in_fan_out) * self.scaling[adapter]
        return output_tensor

    def get_regularization_loss_with_trace(self, adapter: str) -> torch.Tensor:
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        P_2 = self.lora_P_2[adapter]

        # use output statistics
        # Y = torch.matmul(P_2.T, weight_B)  # (d, r)
        # X = torch.matmul(P_2.T, P_2) # (d, d)
        # M = Y.T @ X @ Y # (r, r)
        # AAt = torch.matmul(weight_A, weight_A.T) # (out, out)
        # norm_sq = torch.sum(AAt.T * M)  # tr(A A^T M) = sum((A A^T) * M)
        # norm_sq = norm_sq * (self.scaling[adapter] ** 2)
        # return norm_sq, self.out_features * self.in_features

        # use input statistics
        # \lVert \Delta W_m \rVert^2_F = tr ( (BAP_2P_2^T)^T (BAP_2P_2^T) )
        X = torch.matmul(weight_A, P_2) # (r, k)
        M = torch.matmul(weight_B.T, weight_B)  # (r, r)
        norm_sq = torch.sum(X * (torch.matmul(M, X)))
        norm_sq = norm_sq * (self.scaling[adapter] ** 2)
        return norm_sq, self.out_features * self.in_features

    def get_delta_KpKp(self, adapter: str) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter)
        lora_S_KpKp = self.lora_S_KpKp[adapter]
        delta_KpKp = delta_weight @ lora_S_KpKp @ delta_weight.T
        return delta_KpKp

    def get_previous_loss_with_trace(self, adapter: str) -> tuple[torch.Tensor, int]:
        delta_weight = self.get_delta_weight(adapter)
        lora_S_KpKp = self.lora_S_KpKp[adapter]
        lora_S_KpVp = self.lora_S_KpVp[adapter]
        lora_S_VpVp = self.lora_S_VpVp[adapter]
        weight = self.base_layer.weight + delta_weight

        S_count = self.lora_S_count[adapter]

        # flops identical but peak memory lower
        if self.in_features <= self.out_features:
            # Use the smaller intermediate: compute weight.T @ weight
            # and use elementwise dot for trace(W S_KK W^T)
            K = weight.T @ weight # (in_features, in_features)
            # tr(weight @ S_KpKp @ weight.T) = sum((weight.T @ weight).T * S_KpKp)
            # K is symmetric, so K.T = K
            # but, there may be rounding / accumulation errors, causing K.T \neq K
            trace1 = torch.sum(K.T * lora_S_KpKp)
        else:
            T = weight @ lora_S_KpKp # (out_features, in_features)
            trace1 = torch.sum(T * weight)

        # weight (out_features, in_features)
        # S_KpVp (in_features, out_features)
        # tr(weight @ S_KpVp) = sum(weight * S_KpVp.T) = sum(weight.T * S_KpVp)
        # we use the former to save an extra transpose
        trace2 = 2.0 * torch.sum(weight * lora_S_KpVp.T) # (out_features, in_features)

        trace3 = torch.trace(lora_S_VpVp)

        trace = (trace1 - trace2 + trace3)

        return trace, S_count * self.out_features
        
        # Alternative way (less efficient):
        # trace_KpKp_VpVp = ( torch.trace(weight @ lora_S_KpKp @ weight.T) - (2 * torch.trace(weight @ lora_S_KpVp)) + torch.trace(lora_S_VpVp) ) / self.out_features
        # return trace_KpKp_VpVp

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        alpha = 1.0
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights = orig_weights + alpha * delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1) * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data = base_layer.weight.data + alpha * delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self._get_weight_norm(base_layer.weight, delta_weight, scaling=1).detach()
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        new_weight = dora_factor.view(-1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)


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
