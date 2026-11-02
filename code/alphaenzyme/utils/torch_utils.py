from contextlib import nullcontext
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter





def autocasting_disable_decorator(disable_casting):
    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if disable_casting
                else nullcontext()
            )

            # Helper function to conditionally cast tensors
            def conditioned_cast(tensor):
                if (
                    disable_casting
                    and isinstance(tensor, torch.Tensor)
                    and torch.is_floating_point(tensor)
                ):
                    return tensor.to(dtype=torch.float32)
                return tensor


            # update v2, it can process dict values
            def conditioned_cast_v2(value):
                # 处理容器类型
                if isinstance(value, dict):
                    return {k: conditioned_cast(v) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    return type(value)(conditioned_cast(v) for v in value)

                # 处理张量
                elif (disable_casting and
                      isinstance(value, torch.Tensor) and
                      torch.is_floating_point(value)):
                    return value.to(torch.float32)

                # 其他类型保持不变
                return value

            # use v2 version
            with _amp_context:
                return func(
                    *(conditioned_cast_v2(v) for v in args),
                    **{k: conditioned_cast_v2(v) for k, v in kwargs.items()},
                )

            # with _amp_context:
            #     return func(
            #         *(conditioned_cast(v) for v in args),
            #         **{k: conditioned_cast(v) for k, v in kwargs.items()},
            #     )

        return new_func

    return func_wrapper


def cdist(a: torch.Tensor, b: torch.Tensor = None):
    # for tensor shape [1, 512 * 14, 3], donot_use_mm_for_euclid_dist mode costs 0.0489s,
    # while use_mm_for_euclid_dist_if_necessary costs 0.0419s on cpu. On GPU there two costs
    # will be neglectible. So there is no need to sacrifice accuracy for speed here.
    return torch.cdist(
        a,
        b if b is not None else a,
        compute_mode="donot_use_mm_for_euclid_dist",
    )