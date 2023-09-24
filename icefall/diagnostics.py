# Copyright      2022  Xiaomi Corp.        (authors: Daniel Povey
#                                                    Zengwei Yao
#                                                    Mingshuang Luo)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn


class TensorDiagnosticOptions(object):
    """Options object for tensor diagnostics:

    Args:
      max_eig_dim:
        The maximum dimension for which we print out eigenvalues
        (limited for speed reasons).
    """

    def __init__(self, max_eig_dim: int = 512):
        self.max_eig_dim = max_eig_dim

    def dim_is_summarized(self, size: int):
        return size > 10 and size != 31


def get_tensor_stats(
    x: Tensor,
    dim: int,
    stats_type: str,
) -> Tuple[Tensor, int]:
    """
    Returns the specified transformation of the Tensor (either x or x.abs()
    or (x > 0), summed over all but the index `dim`.

    Args:
      x:
        Tensor, tensor to be analyzed
      dim:
        Dimension with 0 <= dim < x.ndim
      stats_type:
        The stats_type includes several types:
        "abs" -> take abs() before summing
        "positive" -> take (x > 0) before summing
        "rms" -> square before summing, we'll take sqrt later
        "value"  -> just sum x itself
        "max", "min" -> take the maximum or minimum [over all other dims but dim] instead of summing
    Returns:
      stats: a Tensor of shape (x.shape[dim],).
      count: an integer saying how many items were counted in each element
      of stats.
    """

    count = x.numel() // x.shape[dim]

    if stats_type == "eigs":
        x = x.transpose(dim, -1)
        x = x.reshape(-1, x.shape[-1])
        # shape of returned tensor: (s, s),
        # where s is size of dimension `dim` of original x.
        return torch.matmul(x.transpose(0, 1), x), count
    elif stats_type == "abs":
        x = x.abs()
    elif stats_type == "rms":
        x = x**2
    elif stats_type == "positive":
        x = (x > 0).to(dtype=torch.float)
    else:
        assert stats_type in ["value", "max", "min"]

    sum_dims = [d for d in range(x.ndim) if d != dim]
    if len(sum_dims) > 0:
        if stats_type == "max":
            for dim in reversed(sum_dims):
                x = torch.max(x, dim=dim)[0]
        elif stats_type == "min":
            for dim in reversed(sum_dims):
                x = torch.min(x, dim=dim)[0]
        else:
            x = torch.sum(x, dim=sum_dims)
    x = x.flatten().clone()
    return x, count


@dataclass
class TensorAndCount:
    tensor: Tensor
    count: int


class TensorDiagnostic(object):
    """This class is not directly used by the user, it is responsible for
    collecting diagnostics for a module or parameter tensor of a torch.nn.Module.

    Args:
      opts:
        Options object.
      name:
        The name associated with this diagnostics object, will probably be {module_name}.X
           where X is "output" or "grad", or {parameter_name}.Y where Y is param_value or param_grad.
    """

    def __init__(self, opts: TensorDiagnosticOptions, name: str):
        self.opts = opts
        self.name = name
        self.class_name = None  # will assign in accumulate()

        self.stats = None  # we'll later assign a list to self.stats.
        # It's a list of dicts, indexed by dim (i.e. by the
        # axis of the tensor).  The dicts, in turn, are
        # indexed by `stats-type` which are strings in
        # ["abs", "max", "min", "positive", "value", "rms"].

        # scalar_stats contains some analysis of the activations and gradients,
        self.scalar_stats = None

        # the keys into self.stats[dim] are strings, whose values can be
        # "abs", "max", "min" ,"value", "positive", "rms", "value".
        # The values e.g. self.stats[dim]["rms"] are lists of dataclass TensorAndCount,
        # containing a tensor and its associated count (which is the sum of the other dims
        # that we aggregated over, e.g. the number of frames and/or batch elements and/or
        # channels.
        # ... we actually accumulate the Tensors / counts any time we have the same-dim tensor,
        # only adding a new element to the list if there was a different dim.
        # if the string in the key is "eigs", if we detect a length mismatch we put None as the value.

    def accumulate(self, x, class_name: Optional[str] = None):
        """
        Accumulate tensors.
        """
        if class_name is not None:
            self.class_name = class_name
        if isinstance(x, Tuple):
            x = x[0]
        if not isinstance(x, Tensor):
            return
        if x.numel() == 0:  # for empty tensor
            return
        x = x.detach().clone()
        if x.ndim == 0:
            x = x.unsqueeze(0)
        ndim = x.ndim
        if self.stats is None:
            self.stats = [dict() for _ in range(ndim)]

        for dim in range(ndim):
            this_dim_stats = self.stats[dim]
            if ndim > 1:
                stats_types = ["abs", "max", "min", "positive", "value", "rms"]
                if x.shape[dim] <= self.opts.max_eig_dim:
                    stats_types.append("eigs")
            else:
                stats_types = ["value", "abs", "max", "min"]

            for stats_type in stats_types:
                stats, count = get_tensor_stats(x, dim, stats_type)
                if stats_type not in this_dim_stats:
                    this_dim_stats[stats_type] = []  # list of TensorAndCount

                done = False
                if this_dim_stats[stats_type] is None:
                    # we can reach here if we detected for stats_type "eigs" that
                    # where was more than one different size for this dim.  Then we
                    # disable accumulating this stats type, as it uses too much memory.
                    continue
                for s in this_dim_stats[stats_type]:
                    if s.tensor.shape == stats.shape:
                        if stats_type == "max":
                            s.tensor = torch.maximum(s.tensor, stats)

                        elif stats_type == "min":
                            s.tensor = torch.minimum(s.tensor, stats)
                        else:
                            assert stats_type != "max"
                            s.tensor += stats
                        s.count += count
                        done = True
                        break
                if not done:
                    if this_dim_stats[stats_type] != [] and stats_type == "eigs":
                        # >1 size encountered on this dim, e.g. it's a batch or time dimension,
                        # don't accumulat "eigs" stats type, it uses too much memory
                        this_dim_stats[stats_type] = None
                    else:
                        this_dim_stats[stats_type].append(TensorAndCount(stats, count))

    def print_diagnostics(self):
        """Print diagnostics for each dimension of the tensor."""
        if self.stats is None:
            print(f"Warning: the stats of {self.name} is None.")
            return
        for dim, this_dim_stats in enumerate(self.stats):
            if "rms" in this_dim_stats and "value" in this_dim_stats:
                # produce "stddev" stats, which is centered RMS.
                rms_stats_list = this_dim_stats["rms"]
                value_stats_list = this_dim_stats["value"]
                if len(rms_stats_list) == len(value_stats_list):
                    stddev_stats_list = []
                    for r, v in zip(rms_stats_list, value_stats_list):
                        stddev_stats_list.append(
                            # r.count and v.count should be the same, but we don't check this.
                            TensorAndCount(
                                r.tensor - v.tensor * v.tensor / (v.count + 1.0e-20),
                                r.count,
                            )
                        )
                    this_dim_stats["stddev"] = stddev_stats_list

            for stats_type, stats_list in this_dim_stats.items():
                # stats_type could be "rms", "value", "abs", "eigs", "positive", "min" or "max".
                # "stats_list" could be a list of TensorAndCount (one list per distinct tensor
                # shape of the stats), or None
                if stats_list is None:
                    assert stats_type == "eigs"
                    continue

                def get_count(count):
                    return 1 if stats_type in ["max", "min"] else count

                if len(stats_list) == 1:
                    stats = stats_list[0].tensor / get_count(stats_list[0].count)
                else:
                    # a dimension that has variable size in different nnet
                    # forwards, e.g. a time dimension in an ASR model.
                    stats = torch.cat(
                        [x.tensor / get_count(x.count) for x in stats_list], dim=0
                    )

                if stats_type == "eigs":
                    try:
                        eigs, _ = torch.symeig(stats)
                        stats = eigs.abs().sqrt()
                    except:  # noqa
                        print("Error getting eigenvalues, trying another method.")
                        eigs, _ = torch.eig(stats)
                        stats = eigs.norm(dim=1).sqrt()
                        # sqrt so it reflects data magnitude, like stddev- not variance

                if stats_type in ["rms", "stddev"]:
                    # we stored the square; after aggregation we need to take sqrt.
                    stats = stats.sqrt()

                # if `summarize` we print percentiles of the stats; else,
                # we print out individual elements.
                summarize = (len(stats_list) > 1) or self.opts.dim_is_summarized(
                    stats.numel()
                )
                if summarize:  # usually `summarize` will be true
                    # print out percentiles.
                    stats = stats.sort()[0]
                    num_percentiles = 10
                    size = stats.numel()
                    percentiles = []
                    for i in range(num_percentiles + 1):
                        index = (i * (size - 1)) // num_percentiles
                        percentiles.append(stats[index].item())
                    percentiles = ["%.2g" % x for x in percentiles]
                    percentiles = " ".join(percentiles)
                    ans = f"percentiles: [{percentiles}]"
                else:
                    ans = stats.tolist()
                    ans = ["%.2g" % x for x in ans]
                    ans = "[" + " ".join(ans) + "]"
                if stats_type in ["value", "rms", "stddev", "eigs"]:
                    # This norm is useful because it is strictly less than the largest
                    # sqrt(eigenvalue) of the variance, which we print out, and shows,
                    # speaking in an approximate way, how much of that largest eigenvalue
                    # can be attributed to the mean of the distribution.
                    norm = (stats**2).sum().sqrt().item()
                    ans += f", norm={norm:.2g}"
                mean = stats.mean().item()
                rms = (stats**2).mean().sqrt().item()
                ans += f", mean={mean:.3g}, rms={rms:.3g}"

                # OK, "ans" contains the actual stats, e.g.
                # ans = "percentiles: [0.43 0.46 0.48 0.49 0.49 0.5 0.51 0.52 0.53 0.54 0.59], mean=0.5, rms=0.5"

                sizes = [x.tensor.shape[0] for x in stats_list]
                size_str = (
                    f"{sizes[0]}" if len(sizes) == 1 else f"{min(sizes)}..{max(sizes)}"
                )
                maybe_class_name = (
                    f" type={self.class_name}," if self.class_name is not None else ""
                )
                print(
                    f"module={self.name},{maybe_class_name} dim={dim}, size={size_str}, {stats_type} {ans}"
                )


class ScalarDiagnostic(object):
    """This class is not directly used by the user, it is responsible for
    collecting diagnostics for a single module (subclass of torch.nn.Module) that
    represents some kind of nonlinearity, e.g. ReLU, sigmoid, etc.
    """

    def __init__(self, opts: TensorDiagnosticOptions, name: str):
        self.opts = opts
        self.name = name
        self.class_name = None  # will assign in accumulate()
        self.is_forward_pass = True

        self.tick_scale = None

        self.saved_inputs = []
        self.is_ok = True

        self.counts = None
        self.sum_grad = None
        self.sum_gradsq = None
        self.sum_abs_grad = None

    def accumulate_input(self, x: Tensor, class_name: Optional[str] = None):
        """
        Called in forward pass.
        """
        if not self.is_forward_pass:
            # in case we did a forward pass without a backward pass, for some reason.
            self.saved_inputs = []
            self.is_forward_pass = True

        if class_name is not None:
            self.class_name = class_name
        if not self.is_ok:
            return

        limit = 10
        if len(self.saved_inputs) > limit:
            print(
                f"ERROR: forward pass called for this module over {limit} times with no backward pass. "
                f" Will not accumulate scalar stats."
            )
            self.is_ok = False
            return
        self.saved_inputs.append(x)

    def accumulate_output_grad(self, grad: Tensor):
        if not self.is_ok:
            return
        if self.is_forward_pass:
            self.is_forward_pass = False

        last_shape = (
            "n/a" if len(self.saved_inputs) == 0 else self.saved_inputs[-1].shape
        )
        if len(self.saved_inputs) == 0 or grad.shape != last_shape:
            print(
                f"ERROR: shape mismatch or no forward activation present when backward "
                f"pass called: grad shape ={tuple(grad.shape)}, num-saved-inputs={len(self.saved_inputs)}"
                f", shape-of-last-saved-input={last_shape}"
            )
            self.is_ok = False
            return

        x = self.saved_inputs.pop()
        self.process_input_and_grad(x, grad)

    def process_input_and_grad(self, x: Tensor, grad: Tensor):
        assert x.shape == grad.shape
        x = x.flatten()
        grad = grad.flatten()

        num_ticks_per_side = 256

        if self.tick_scale is None:
            x_abs_sorted = x.abs().sort()[0]
            # take the 98th percentile as the largest value we count separately.
            index = int(x.numel() * 0.98)
            self.tick_scale = float(x_abs_sorted[index] / num_ticks_per_side)

            # integerize from tick * (-num ticks_per_side ..  num_ticks_per_side - 1]
            self.counts = torch.zeros(
                2 * num_ticks_per_side, dtype=torch.long, device=x.device
            )
            self.sum_grad = torch.zeros(
                2 * num_ticks_per_side, dtype=torch.double, device=x.device
            )
            # sum_gradsq is for getting error bars.
            self.sum_gradsq = torch.zeros(
                2 * num_ticks_per_side, dtype=torch.double, device=x.device
            )
            self.sum_abs_grad = torch.zeros(
                2 * num_ticks_per_side, dtype=torch.double, device=x.device
            )

        # this will round down.
        x = (x / self.tick_scale).to(torch.long)
        x = x.clamp_(min=-num_ticks_per_side, max=num_ticks_per_side - 1)
        x = x + num_ticks_per_side

        self.counts.index_add_(dim=0, index=x, source=torch.ones_like(x))
        self.sum_grad.index_add_(dim=0, index=x, source=grad.to(torch.double))
        self.sum_gradsq.index_add_(
            dim=0, index=x, source=(grad * grad).to(torch.double)
        )
        self.sum_abs_grad.index_add_(dim=0, index=x, source=grad.abs().to(torch.double))

    def print_diagnostics(self):
        """Print diagnostics."""
        if self.is_ok is False or self.counts is None:
            print(f"Warning: no stats accumulated for {self.name}, is_ok={self.is_ok}")
            return

        counts = self.counts.to("cpu")
        sum_grad = self.sum_grad.to(device="cpu", dtype=torch.float32)
        sum_gradsq = self.sum_gradsq.to(device="cpu", dtype=torch.float32)
        sum_abs_grad = self.sum_abs_grad.to(device="cpu", dtype=torch.float32)

        counts_cumsum = counts.cumsum(dim=0)
        counts_tot = counts_cumsum[-1]

        # subdivide the distribution up into `num_bins` intervals for analysis, for greater
        # statistical significance.  each bin corresponds to multiple of the original 'tick' intervals.
        num_bins = 20

        # integer division
        counts_per_bin = (counts_tot // num_bins) + 1
        bin_indexes = counts_cumsum // counts_per_bin
        bin_indexes = bin_indexes.clamp(min=0, max=num_bins).to(torch.long)

        bin_counts = torch.zeros(num_bins, dtype=torch.long)
        bin_counts.index_add_(dim=0, index=bin_indexes, source=counts)
        bin_grad = torch.zeros(num_bins)
        bin_grad.index_add_(dim=0, index=bin_indexes, source=sum_grad)
        bin_gradsq = torch.zeros(num_bins)
        bin_gradsq.index_add_(dim=0, index=bin_indexes, source=sum_gradsq)
        bin_abs_grad = torch.zeros(num_bins)
        bin_abs_grad.index_add_(dim=0, index=bin_indexes, source=sum_abs_grad)

        avg_grad = bin_grad / bin_counts
        avg_grad_stddev = (bin_gradsq / bin_counts).sqrt()

        bin_boundary_counts = (
            torch.arange(num_bins + 1, dtype=torch.long) * counts_per_bin
        )
        bin_tick_indexes = torch.searchsorted(counts_cumsum, bin_boundary_counts)
        # boundaries are the "x" values between the bins, e.g. corresponding to the
        # locations of percentiles of the distribution.
        num_ticks_per_side = counts.numel() // 2
        bin_boundaries = (bin_tick_indexes - num_ticks_per_side) * self.tick_scale

        bin_grad = bin_grad / (bin_counts + 1)
        bin_conf_interval = bin_gradsq.sqrt() / (
            bin_counts + 1
        )  # consider this a standard deviation.
        # bin_grad / bin_abs_grad will give us a sense for how important in a practical sense,
        # the gradients are.
        bin_abs_grad = bin_abs_grad / (bin_counts + 1)

        bin_rel_grad = bin_grad / (bin_abs_grad + 1.0e-20)
        bin_conf = bin_grad / (bin_conf_interval + 1.0e-20)

        def tensor_to_str(x: Tensor):
            x = ["%.2g" % f for f in x]
            x = "[" + " ".join(x) + "]"
            return x

        maybe_class_name = (
            f" type={self.class_name}," if self.class_name is not None else ""
        )

        print(
            f"module={self.name},{maybe_class_name} bin-boundaries={tensor_to_str(bin_boundaries)}, "
            f"rel_grad={tensor_to_str(bin_rel_grad)}, grad_conf={tensor_to_str(bin_conf)}"
        )


class ModelDiagnostic(object):
    """This class stores diagnostics for all tensors in the torch.nn.Module.

    Args:
      opts:
        Options object.
    """

    def __init__(self, opts: Optional[TensorDiagnosticOptions] = None):
        # In this dictionary, the keys are tensors names and the values
        # are corresponding TensorDiagnostic objects.
        if opts is None:
            self.opts = TensorDiagnosticOptions()
        else:
            self.opts = opts
        self.diagnostics = dict()

    def __getitem__(self, name: str):
        T = ScalarDiagnostic if name[-7:] == ".scalar" else TensorDiagnostic
        if name not in self.diagnostics:
            self.diagnostics[name] = T(self.opts, name)
        return self.diagnostics[name]

    def print_diagnostics(self):
        """Print diagnostics for each tensor."""
        for k in sorted(self.diagnostics.keys()):
            self.diagnostics[k].print_diagnostics()


def get_class_name(module: nn.Module):
    ans = type(module).__name__
    # we put the below in try blocks in case anyone is using a different version of these modules that
    # might have different member names.
    if ans == "Balancer" or ans == "ActivationBalancer":
        try:
            ans += f"[{float(module.min_positive)},{float(module.max_positive)},{float(module.min_abs)},{float(module.max_abs)}]"
        except:
            pass
    elif ans == "AbsValuePenalizer":
        try:
            ans += f"[{module.limit}]"
        except:
            pass
    return ans


def attach_diagnostics(
    model: nn.Module, opts: Optional[TensorDiagnosticOptions] = None
) -> ModelDiagnostic:
    """Attach a ModelDiagnostic object to the model by
    1) registering forward hook and backward hook on each module, to accumulate
    its output tensors and gradient tensors, respectively;
    2) registering backward hook on each module parameter, to accumulate its
    values and gradients.

    Args:
      model:
        the model to be analyzed.
      opts:
        Options object.

    Returns:
      The ModelDiagnostic object attached to the model.
    """

    ans = ModelDiagnostic(opts)
    for name, module in model.named_modules():
        if name == "":
            name = "<top-level>"

        # Setting model_diagnostic=ans and n=name below, instead of trying to
        # capture the variables, ensures that we use the current values.
        # (this matters for `name`, since the variable gets overwritten).
        # These closures don't really capture by value, only by
        # "the final value the variable got in the function" :-(
        def forward_hook(_module, _input, _output, _model_diagnostic=ans, _name=name):
            if isinstance(_output, tuple) and len(_output) == 1:
                _output = _output[0]

            if isinstance(_output, Tensor) and _output.dtype in (
                torch.float32,
                torch.float16,
                torch.float64,
            ):
                _model_diagnostic[f"{_name}.output"].accumulate(
                    _output, class_name=get_class_name(_module)
                )
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    if o.dtype in (torch.float32, torch.float16, torch.float64):
                        _model_diagnostic[f"{_name}.output[{i}]"].accumulate(
                            o, class_name=get_class_name(_module)
                        )

        def backward_hook(_module, _input, _output, _model_diagnostic=ans, _name=name):
            if isinstance(_output, tuple) and len(_output) == 1:
                _output = _output[0]
            if isinstance(_output, Tensor) and _output.dtype in (
                torch.float32,
                torch.float16,
                torch.float64,
            ):
                _model_diagnostic[f"{_name}.grad"].accumulate(
                    _output, class_name=get_class_name(_module)
                )
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    if o.dtype in (torch.float32, torch.float16, torch.float64):
                        _model_diagnostic[f"{_name}.grad[{i}]"].accumulate(
                            o, class_name=get_class_name(_module)
                        )

        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

        if type(module).__name__ in [
            "Sigmoid",
            "Tanh",
            "ReLU",
            "TanSwish",
            "Swish",
            "DoubleSwish",
            "Swoosh",
        ]:
            # For these specific module types, accumulate some additional diagnostics
            # that can help us improve the activation function.  These require a lot of memory,
            # to save the forward activations, so limit this to some select classes.
            # Note: this will not work correctly for all model types.
            def scalar_forward_hook(
                _module, _input, _output, _model_diagnostic=ans, _name=name
            ):
                if isinstance(_input, tuple):
                    (_input,) = _input
                assert isinstance(_input, Tensor)
                _model_diagnostic[f"{_name}.scalar"].accumulate_input(
                    _input, class_name=get_class_name(_module)
                )

            def scalar_backward_hook(
                _module, _input, _output, _model_diagnostic=ans, _name=name
            ):
                if isinstance(_output, tuple):
                    (_output,) = _output
                assert isinstance(_output, Tensor)
                _model_diagnostic[f"{_name}.scalar"].accumulate_output_grad(_output)

            module.register_forward_hook(scalar_forward_hook)
            module.register_backward_hook(scalar_backward_hook)

    for name, parameter in model.named_parameters():

        def param_backward_hook(
            grad, _parameter=parameter, _model_diagnostic=ans, _name=name
        ):
            _model_diagnostic[f"{_name}.param_value"].accumulate(_parameter)
            _model_diagnostic[f"{_name}.param_grad"].accumulate(grad)

        parameter.register_hook(param_backward_hook)

    return ans


def _test_tensor_diagnostic():
    opts = TensorDiagnosticOptions(512)

    diagnostic = TensorDiagnostic(opts, "foo")

    for _ in range(10):
        diagnostic.accumulate(torch.randn(50, 100) * 10.0)

    diagnostic.print_diagnostics()

    model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 80))

    diagnostic = attach_diagnostics(model, opts)
    for _ in range(10):
        T = random.randint(200, 300)
        x = torch.randn(T, 100)
        y = model(x)
        y.sum().backward()

    diagnostic.print_diagnostics()


if __name__ == "__main__":
    _test_tensor_diagnostic()
