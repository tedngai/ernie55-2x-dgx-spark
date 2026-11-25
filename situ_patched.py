"""Kimi K3 SiTU wrappers for the activation-specialized fused MoE backends.

The shared backend implements ``beta * tanh(gate / beta) * sigmoid(gate)``
times ``linear_beta * tanh(up / linear_beta)`` with K3's beta values.
"""

from __future__ import annotations

from typing import Tuple

from b12x.moe._shared.kernels.activations import SITU
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from b12x.moe._shared.kernels.micro import MoEMicroKernelBackend


class MoEMicroKernelSitu(MoEMicroKernelBackend):
    """SiTU micro specialization, trellis-only.

    The compact NVFP4 intermediate quantization has not passed its
    correctness gate, so ``is_supported`` stays False and the nvfp4-family
    dispatch never selects this class. The trellis_t256 arm quantizes its
    intermediate through per-32 UE8M0 E4M3 (a8_mx) instead; the w4a8_mx
    trellis band dispatches it explicitly and constructs this class with
    ``weight_layout="trellis_t256"``.
    """

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        del m, k, n, num_topk, weight_E
        return False

    def __init__(self, *args: object, **kwargs: object):
        if kwargs.get("weight_layout") != "trellis_t256":
            raise NotImplementedError(
                "SiTU compact micro serves only the trellis_t256 weight "
                "layout; select the fused dynamic kernel"
            )
        kwargs["activation"] = SITU
        super().__init__(*args, **kwargs)


class MoEDynamicKernelSitu(MoEDynamicKernelBackend):
    def __init__(self, *args: object, **kwargs: object) -> None:
        # Forward every shared-backend option (W4A8 layout flags, routing
        # plans, ...) like the SiLU specialization does; only the activation
        # differs. b12x 1.5.1's explicit signature missed the W4A8 kwargs.
        kwargs["activation"] = SITU
        super().__init__(*args, **kwargs)


__all__ = ["MoEDynamicKernelSitu", "MoEMicroKernelSitu"]
