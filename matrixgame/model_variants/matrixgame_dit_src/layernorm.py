
import importlib
import inspect
import numbers

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

try:
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
except ImportError:
    FusedRMSNormAffineMixedDtypesFunction = None


global fused_layer_norm_cuda
fused_layer_norm_cuda = None

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

    HAVE_PERSIST_LAYER_NORM = True
except ImportError:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    HAVE_FUSED_LAYER_NORM = False

from apex.normalization.fused_layer_norm import fused_layer_norm


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad)
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        """Runs the fwd pass of _kernel_make_viewless_tensor"""
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        """No-op"""
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)
    

class FusedLayerNorm(torch.nn.Module):
    """Layer Norm, fused into a single CUDA kernel.

    Args:
      hidden_size (int): Transformer hidden dimension.

      eps (float): Epsilon added to denominator, for numerical stability.

      persist_layer_norm (bool): Use persistent fused layer norm kernel.
      This kernel supports only a set of hidden sizes. Please
      check persist_ln_hidden_sizes if your hidden size is supported.

      zero_centered_gamma (bool): Adjust LayerNorm weights such that they are
      centered around zero. This improves numerical stability.

      config (TransformerConfig): Transformer config. Include to match custom
      layer norm interfaces.

      normalization (str): Normalization type, used for Transformer Engine.
      Must equal 'LayerNorm' here.
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = False, 
        eps: float = 1e-5,
        persist_layer_norm: bool = True,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",  # included to match TE interface
        memory_efficient : bool = False
    ):
        super().__init__()

        self.zero_centered_gamma = zero_centered_gamma
        

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if hidden_size not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:
            persist_layer_norm = False

        if not persist_layer_norm and not HAVE_FUSED_LAYER_NORM:
            # TODO: Add pytorch only layer norm
            raise ValueError(f'Apex must be installed to use FusedLayerNorm.')

        if isinstance(hidden_size, numbers.Integral):
            hidden_size = (hidden_size,)
        self.hidden_size = torch.Size(hidden_size)
        self.normalized_shape = torch.Size(hidden_size)
        self.memory_efficient = memory_efficient
        self.eps = eps
        # Parameters need to be initialized with torch.empty rather than torch.Tensor for correct device placement with nemo2.
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(torch.empty(*hidden_size))
            self.bias = Parameter(torch.empty(*hidden_size))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.persist_layer_norm = persist_layer_norm

    # def reset_parameters(self):

    #     if self.zero_centered_gamma:
    #         init.zeros_(self.weight)
    #         init.zeros_(self.bias)
    #     else:
    #         init.ones_(self.weight)
    #         init.zeros_(self.bias)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            if self.zero_centered_gamma:
                init.zeros_(self.weight)
            else:
                init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)    

    def forward(self, input: Tensor) -> Tensor:

        if not self.elementwise_affine:
            return fused_layer_norm(input, self.normalized_shape, self.eps, self.memory_efficient)
        
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight

        if self.persist_layer_norm:
            if 'memory_efficient' in inspect.getfullargspec(FastLayerNormFN.forward).args:
                output = FastLayerNormFN.apply(
                    input, weight, self.bias, self.eps, False
                )
            else:
                output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(
                inp=output, requires_grad=input.requires_grad, keep_graph=True
            )

        else:
            if (
                'memory_efficient'
                in inspect.getfullargspec(FusedLayerNormAffineFunction.forward).args
            ):
                return FusedLayerNormAffineFunction.apply(
                    input,
                    weight,
                    self.bias,
                    self.hidden_size,
                    self.eps,
                    False,
                )
            else:
                return FusedLayerNormAffineFunction.apply(
                    input, weight, self.bias, self.hidden_size, self.eps
                )

        return output
    

class MixedFusedRMSNorm(torch.nn.Module):

  def __init__(self, 
               normalized_shape, 
               elementwise_affine=True,
               eps=1e-5,
               device = None, 
               dtype = None):
        super(MixedFusedRMSNorm, self).__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")


        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

        # set sequence parallelism flag on weight and bias parameters
        # setattr(self.weight, 'sequence_parallel', sequence_parallel)

  def reset_parameters(self):
    init.ones_(self.weight)

  def forward(self, input):
    try:
      try:
        return FusedRMSNormAffineMixedDtypesFunction.apply(input, self.weight, self.normalized_shape, self.eps, True)
      except:
        return FusedRMSNormAffineMixedDtypesFunction.apply(input, self.weight, self.normalized_shape, self.eps)
    except:
      # `memory_efficient=False` aligns with the old version
      # `apply` doesn't accept keyword arguments
      return FusedRMSNormAffineMixedDtypesFunction.apply(input, self.weight, self.normalized_shape, self.eps, False)


if __name__ == "__main__":
    norm = FusedLayerNorm(3072).to("cuda")
    x = torch.rand(1,  21730, 3072, dtype=torch.float32, device="cuda")
    
    y = norm(x)
    
    
    torch.cuda.synchronize()
    import time
    N = 100
    start = time.time()
    for i in range(N):
        y1 = norm(x)
    torch.cuda.synchronize()
    
    cost = (time.time() - start) / N
    print(f"apex layernorm cost: {cost}")
    
    
    from torch import nn
    start = time.time()
    layer_norm = nn.LayerNorm(3072).to("cuda")
    for i  in range(N):
        y2 = layer_norm(x)
        
    torch.cuda.synchronize()
    
    print(y1)
    print(y2)
    
    print(f"equal: {torch.isclose(y1, y2, rtol=1e-5, atol=1e-8)}")
    
    cost = (time.time() - start) / N
    print(f"torch layernorm cost: {cost}")
        