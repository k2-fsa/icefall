# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This is modified from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Optional
from collections import OrderedDict
import numpy as np

Tensor = torch.Tensor

module_flop_count = []
old_functions = {}


class FlopsProfiler(object):
    """Measures the latency, number of estimated floating-point operations and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how latency, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated latency, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input.

    To profile a trained model in inference, use the `get_model_profile` API.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """

    def __init__(self, model, module_hoop_mapping=None):
        self.model = model
        self.started = False
        self.func_patched = False
        self.module_hoop_mapping = (
            module_hoop_mapping
            if module_hoop_mapping is not None
            else MODULE_HOOK_MAPPING
        )

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_functionals()
        _patch_tensor_methods()

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in self.module_hoop_mapping:
                if not hasattr(module, "__flops_handle__"):
                    module.__flops_handle__ = module.register_forward_hook(
                        self.module_hoop_mapping[type(module)]
                    )
                return

            # if computing the flops of the functionals in a module
            def pre_hook(module, input):
                module_flop_count.append([])

            if not hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                if module_flop_count:
                    module.__flops__ += sum([elem[1] for elem in module_flop_count[-1]])
                    module_flop_count.pop()

            if not hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)

        self.model.apply(partial(register_module_hooks, ignore_list=ignore_list))
        self.started = True
        self.func_patched = True

    def stop_profile(self):
        """Stop profiling.

        All torch.nn.functionals are restored to their originals.
        """
        if self.started and self.func_patched:
            _reload_functionals()
            _reload_tensor_methods()
            self.func_patched = False

        def remove_profile_attrs(module):
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__

        self.model.apply(remove_profile_attrs)

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """

        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters())

        self.model.apply(add_or_reset_attrs)

    def end_profile(self):
        """Ends profiling.

        The added attributes and handles are removed recursively on all the modules.
        """
        if not self.started:
            return
        self.stop_profile()
        self.started = False

        def remove_profile_attrs(module):
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__params__"):
                del module.__params__

        self.model.apply(remove_profile_attrs)

    def get_total_flops(self, as_string=False):
        """Returns the total flops of the model.

        Args:
            as_string (bool, optional): whether to output the flops as string. Defaults to False.

        Returns:
            The number of multiply-accumulate operations of the model forward pass.
        """
        total_flops = get_module_flops(self.model)
        return num_to_string(total_flops) if as_string else total_flops

    def get_total_params(self, as_string=False):
        """Returns the total parameters of the model.

        Args:
            as_string (bool, optional): whether to output the parameters as string. Defaults to False.

        Returns:
            The number of parameters in the model.
        """
        return (
            params_to_string(self.model.__params__)
            if as_string
            else self.model.__params__
        )


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    macs = input.numel() * out_features
    return 2 * macs


def _relu_flops_compute(input, inplace=False):
    return input.numel()


def _prelu_flops_compute(input: Tensor, weight: Tensor):
    return input.numel()


def _elu_flops_compute(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return input.numel()


def _leaky_relu_flops_compute(
    input: Tensor, negative_slope: float = 0.01, inplace: bool = False
):
    return input.numel()


def _relu6_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel()


def _silu_flops_compute(input: Tensor, inplace: bool = False):
    return input.numel()


def _gelu_flops_compute(input, **kwargs):
    return input.numel()


def _pool_flops_compute(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=None,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    return_indices=None,
):
    return input.numel()


def _conv_flops_compute(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (
            input_dim
            + 2 * paddings[idx]
            - (dilations[idx] * (kernel_dims[idx] - 1) + 1)
        ) // strides[idx] + 1
        output_dims.append(output_dim)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    return int(overall_conv_flops + bias_flops)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[2:])
    input_dims = list(input.shape[2:])

    length = len(input_dims)

    paddings = padding if type(padding) is tuple else (padding,) * length
    strides = stride if type(stride) is tuple else (stride,) * length
    dilations = dilation if type(dilation) is tuple else (dilation,) * length

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (
            input_dim
            + 2 * paddings[idx]
            - (dilations[idx] * (kernel_dims[idx] - 1) + 1)
        ) // strides[idx] + 1
        output_dims.append(output_dim)

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_macs = conv_per_position_macs * active_elements_count
    overall_conv_flops = 2 * overall_conv_macs

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    return int(overall_conv_flops + bias_flops)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    has_affine = weight is not None
    if training:
        # estimation
        return input.numel() * (5 if has_affine else 4), 0
    flops = input.numel() * (2 if has_affine else 1)
    return flops


def _layer_norm_flops_compute(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4)


def _group_norm_flops_compute(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4)


def _instance_norm_flops_compute(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    has_affine = weight is not None
    # estimation
    return input.numel() * (5 if has_affine else 4)


def _upsample_flops_compute(input, **kwargs):
    size = kwargs.get("size", None)
    if size is not None:
        if isinstance(size, tuple) or isinstance(size, list):
            return int(_prod(size)), 0
        else:
            return int(size), 0
    scale_factor = kwargs.get("scale_factor", None)
    assert scale_factor is not None, "either size or scale_factor should be defined"
    flops = input.numel()
    if isinstance(scale_factor, tuple) and len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor ** len(input)
    return flops


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return input.numel()


def _sigmoid_flops_compute(input):
    return input.numel()


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0


def _matmul_flops_compute(input, other, *, out=None):
    """
    Count flops for the matmul operation.
    """
    macs = _prod(input.shape) * other.shape[-1]
    return 2 * macs


def _addmm_flops_compute(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(input.shape)


def _einsum_flops_compute(equation, *operands):
    """
    Count flops for the einsum operation.
    """
    equation = equation.replace(" ", "")
    input_shapes = [o.shape for o in operands]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    np_arrs = [np.zeros(s) for s in input_shapes]
    optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
    for line in optim.split("\n"):
        if "optimized flop" in line.lower():
            flop = int(float(line.split(":")[-1]))
            return flop
    raise NotImplementedError("Unsupported einsum operation.")


def _tensor_addmm_flops_compute(self, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    Count flops for the tensor addmm operation.
    """
    macs = _prod(mat1.shape) * mat2.shape[-1]
    return 2 * macs + _prod(self.shape)


def _mul_flops_compute(input, other, *, out=None):
    print("mul")
    return _elementwise_flops_compute(input, other)


def _add_flops_compute(input, other, *, alpha=1, out=None):
    print("add")
    return _elementwise_flops_compute(input, other)


def _sum_flops_compute(input, dim, keepdim=False):
    return input.numel()


def _elementwise_flops_compute(input, other):
    if not torch.is_tensor(input):
        if torch.is_tensor(other):
            return _prod(other.shape)
        else:
            return 1
    elif not torch.is_tensor(other):
        return _prod(input.shape)
    else:
        dim_input = len(input.shape)
        dim_other = len(other.shape)
        max_dim = max(dim_input, dim_other)

        final_shape = []
        for i in range(max_dim):
            in_i = input.shape[i] if i < dim_input else 1
            ot_i = other.shape[i] if i < dim_other else 1
            if in_i > ot_i:
                final_shape.append(in_i)
            else:
                final_shape.append(ot_i)
        flops = _prod(final_shape)
        return flops


def _tanh_flops_compute(input):
    return input.numel()


def _k2_swoosh_flops_compute(input):
    # For SwooshLForward and SwooshRForward
    # estimate as swish/silu
    return input.numel()


def wrapFunc(func, funcFlopCompute):
    oldFunc = func
    name = func.__str__
    old_functions[name] = oldFunc

    def newFunc(*args, **kwds):
        flops = funcFlopCompute(*args, **kwds)
        if module_flop_count:
            module_flop_count[-1].append((name, flops))
        return oldFunc(*args, **kwds)

    newFunc.__str__ = func.__str__

    return newFunc


def _patch_functionals():
    # FC
    F.linear = wrapFunc(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute)

    # activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = wrapFunc(F.prelu, _prelu_flops_compute)
    F.elu = wrapFunc(F.elu, _elu_flops_compute)
    F.leaky_relu = wrapFunc(F.leaky_relu, _leaky_relu_flops_compute)
    F.relu6 = wrapFunc(F.relu6, _relu6_flops_compute)
    if hasattr(F, "silu"):
        F.silu = wrapFunc(F.silu, _silu_flops_compute)
    F.gelu = wrapFunc(F.gelu, _gelu_flops_compute)

    # Normalizations
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute)
    F.layer_norm = wrapFunc(F.layer_norm, _layer_norm_flops_compute)
    F.instance_norm = wrapFunc(F.instance_norm, _instance_norm_flops_compute)
    F.group_norm = wrapFunc(F.group_norm, _group_norm_flops_compute)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _pool_flops_compute)

    # upsample
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute)

    # softmax
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute)

    # sigmoid
    F.sigmoid = wrapFunc(F.sigmoid, _sigmoid_flops_compute)

    # embedding
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute)

    # swoosh functions in k2
    k2.swoosh_l_forward = wrapFunc(k2.swoosh_l_forward, _k2_swoosh_flops_compute)
    k2.swoosh_r_forward = wrapFunc(k2.swoosh_r_forward, _k2_swoosh_flops_compute)
    k2.swoosh_l = wrapFunc(k2.swoosh_l, _k2_swoosh_flops_compute)
    k2.swoosh_r = wrapFunc(k2.swoosh_r, _k2_swoosh_flops_compute)


def _patch_tensor_methods():
    torch.matmul = wrapFunc(torch.matmul, _matmul_flops_compute)
    torch.Tensor.matmul = wrapFunc(torch.Tensor.matmul, _matmul_flops_compute)
    torch.mm = wrapFunc(torch.mm, _matmul_flops_compute)
    torch.Tensor.mm = wrapFunc(torch.Tensor.mm, _matmul_flops_compute)
    torch.bmm = wrapFunc(torch.bmm, _matmul_flops_compute)
    torch.Tensor.bmm = wrapFunc(torch.Tensor.bmm, _matmul_flops_compute)

    torch.addmm = wrapFunc(torch.addmm, _addmm_flops_compute)
    torch.Tensor.addmm = wrapFunc(torch.Tensor.addmm, _tensor_addmm_flops_compute)

    torch.mul = wrapFunc(torch.mul, _mul_flops_compute)
    torch.Tensor.mul = wrapFunc(torch.Tensor.mul, _mul_flops_compute)

    torch.add = wrapFunc(torch.add, _add_flops_compute)
    torch.Tensor.add = wrapFunc(torch.Tensor.add, _add_flops_compute)

    torch.sum = wrapFunc(torch.sum, _sum_flops_compute)
    torch.Tensor.sum = wrapFunc(torch.Tensor.sum, _sum_flops_compute)

    torch.einsum = wrapFunc(torch.einsum, _einsum_flops_compute)

    torch.baddbmm = wrapFunc(torch.baddbmm, _tensor_addmm_flops_compute)

    torch.tanh = wrapFunc(torch.tanh, _tanh_flops_compute)

    torch.Tensor.softmax = wrapFunc(torch.Tensor.softmax, _softmax_flops_compute)

    torch.sigmoid = wrapFunc(torch.sigmoid, _sigmoid_flops_compute)
    torch.Tensor.sigmoid = wrapFunc(torch.Tensor.sigmoid, _sigmoid_flops_compute)


def _reload_functionals():
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions[F.linear.__str__]
    F.conv1d = old_functions[F.conv1d.__str__]
    F.conv2d = old_functions[F.conv2d.__str__]
    F.conv3d = old_functions[F.conv3d.__str__]
    F.conv_transpose1d = old_functions[F.conv_transpose1d.__str__]
    F.conv_transpose2d = old_functions[F.conv_transpose2d.__str__]
    F.conv_transpose3d = old_functions[F.conv_transpose3d.__str__]
    F.relu = old_functions[F.relu.__str__]
    F.prelu = old_functions[F.prelu.__str__]
    F.elu = old_functions[F.elu.__str__]
    F.leaky_relu = old_functions[F.leaky_relu.__str__]
    F.relu6 = old_functions[F.relu6.__str__]
    if hasattr(F, "silu"):
        F.silu = old_functions[F.silu.__str__]
    F.gelu = old_functions[F.gelu.__str__]
    F.batch_norm = old_functions[F.batch_norm.__str__]
    F.layer_norm = old_functions[F.layer_norm.__str__]
    F.instance_norm = old_functions[F.instance_norm.__str__]
    F.group_norm = old_functions[F.group_norm.__str__]
    F.avg_pool1d = old_functions[F.avg_pool1d.__str__]
    F.avg_pool2d = old_functions[F.avg_pool2d.__str__]
    F.avg_pool3d = old_functions[F.avg_pool3d.__str__]
    F.max_pool1d = old_functions[F.max_pool1d.__str__]
    F.max_pool2d = old_functions[F.max_pool2d.__str__]
    F.max_pool3d = old_functions[F.max_pool3d.__str__]
    F.adaptive_avg_pool1d = old_functions[F.adaptive_avg_pool1d.__str__]
    F.adaptive_avg_pool2d = old_functions[F.adaptive_avg_pool2d.__str__]
    F.adaptive_avg_pool3d = old_functions[F.adaptive_avg_pool3d.__str__]
    F.adaptive_max_pool1d = old_functions[F.adaptive_max_pool1d.__str__]
    F.adaptive_max_pool2d = old_functions[F.adaptive_max_pool2d.__str__]
    F.adaptive_max_pool3d = old_functions[F.adaptive_max_pool3d.__str__]
    F.upsample = old_functions[F.upsample.__str__]
    F.interpolate = old_functions[F.interpolate.__str__]
    F.softmax = old_functions[F.softmax.__str__]
    F.sigmoid = old_functions[F.sigmoid.__str__]
    F.embedding = old_functions[F.embedding.__str__]
    # swoosh functions in k2
    k2.swoosh_l = old_functions[k2.swoosh_l.__str__]
    k2.swoosh_r = old_functions[k2.swoosh_r.__str__]
    k2.swoosh_l_forward = old_functions[k2.swoosh_l_forward.__str__]
    k2.swoosh_r_forward = old_functions[k2.swoosh_r_forward.__str__]


def _reload_tensor_methods():
    torch.matmul = old_functions[torch.matmul.__str__]
    torch.Tensor.matmul = old_functions[torch.Tensor.matmul.__str__]
    torch.mm = old_functions[torch.mm.__str__]
    torch.Tensor.mm = old_functions[torch.Tensor.mm.__str__]
    torch.bmm = old_functions[torch.matmul.__str__]
    torch.Tensor.bmm = old_functions[torch.Tensor.bmm.__str__]
    torch.addmm = old_functions[torch.addmm.__str__]
    torch.Tensor.addmm = old_functions[torch.Tensor.addmm.__str__]
    torch.mul = old_functions[torch.mul.__str__]
    torch.Tensor.mul = old_functions[torch.Tensor.mul.__str__]
    torch.add = old_functions[torch.add.__str__]
    torch.Tensor.add = old_functions[torch.Tensor.add.__str__]
    torch.sum = old_functions[torch.sum.__str__]
    torch.Tensor.sum = old_functions[torch.Tensor.sum.__str__]

    torch.einsum = old_functions[torch.einsum.__str__]

    torch.baddbmm = old_functions[torch.baddbmm.__str__]

    torch.Tensor.softmax = old_functions[torch.Tensor.softmax.__str__]

    torch.sigmoid = old_functions[torch.sigmoid.__str__]
    torch.Tensor.sigmoid = old_functions[torch.Tensor.sigmoid.__str__]


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += (
            rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        )
        # final hadamard
        flops += (
            rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        )
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}


def num_to_string(num, precision=2):
    if num // 10**9 > 0:
        return str(round(num / 10.0**9, precision)) + " G"
    elif num // 10**6 > 0:
        return str(round(num / 10.0**6, precision)) + " M"
    elif num // 10**3 > 0:
        return str(round(num / 10.0**3, precision)) + " K"
    else:
        return str(num)


def number_to_string(num, units=None, precision=2):
    if units is None:
        if num // 10**9 > 0:
            return str(round(num / 10.0**9, precision)) + " G"
        elif num // 10**6 > 0:
            return str(round(num / 10.0**6, precision)) + " M"
        elif num // 10**3 > 0:
            return str(round(num / 10.0**3, precision)) + " K"
        else:
            return str(num) + " "
    else:
        if units == "G":
            return str(round(num / 10.0**9, precision)) + " " + units
        elif units == "M":
            return str(round(num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(num / 10.0**3, precision)) + " " + units
        else:
            return str(num) + " "


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**12 > 0:
            return str(round(flops / 10.0**12, precision)) + " TFLOPS"
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GFLOPS"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MFLOPS"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KFLOPS"
        else:
            return str(flops) + " FLOPS"
    else:
        if units == "TFLOPS":
            return str(round(flops / 10.0**12, precision)) + " " + units
        if units == "GFLOPS":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MFLOPS":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KFLOPS":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " FLOPS"


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + " M"
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + " k"
        else:
            return str(params_num)
    else:
        if units == "M":
            return str(round(params_num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(params_num / 10.0**3, precision)) + " " + units
        else:
            return str(params_num)


def get_module_flops(module):
    sum = module.__flops__
    # iterate over immediate children modules
    for child in module.children():
        sum += get_module_flops(child)
    return sum


def get_module_duration(module):
    duration = module.__duration__
    if duration == 0:  # e.g. ModuleList
        for m in module.children():
            duration += m.__duration__
    return duration


def get_model_profile(
    model,
    args=[],
    as_string=True,
    ignore_modules=None,
    module_hoop_mapping=None,
):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, params = get_model_profile(model=model, args=(feature, feature_lens))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        args (list): list of positional arguments to the model.
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    prof = FlopsProfiler(model, module_hoop_mapping=module_hoop_mapping)
    model.eval()

    assert len(args) > 0, "input args must be specified"

    prof.start_profile(ignore_list=ignore_modules)

    _ = model(*args)

    flops = prof.get_total_flops()
    params = prof.get_total_params()

    prof.end_profile()
    if as_string:
        return (
            number_to_string(flops),
            params_to_string(params),
        )

    return flops, params
