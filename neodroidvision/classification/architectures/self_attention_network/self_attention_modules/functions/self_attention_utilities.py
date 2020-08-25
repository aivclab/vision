from collections import namedtuple
from string import Template
from typing import Any

import cupy
import torch

__all__ = ["Stream", "get_dtype_str", "load_kernel"]

Stream = namedtuple("Stream", ["ptr"])


def get_dtype_str(t: torch.Tensor) -> str:
    if isinstance(t, torch.cuda.FloatTensor):
        return "float"
    elif isinstance(t, torch.cuda.DoubleTensor):
        return "double"
    raise NotImplemented(f"Tensor type {t} not supported")


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name: Any, code: str, **kwargs) -> Any:
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N: int) -> int:
    """

:param N:
:type N:
:return:
:rtype:
"""
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


kernel_loop = """
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
"""
