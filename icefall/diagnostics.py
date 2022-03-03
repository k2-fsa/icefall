import random
from typing import List, Tuple

import torch
from torch import Tensor, nn


class TensorDiagnosticOptions(object):
    """Options object for tensor diagnostics:

    Args:
      memory_limit:
        The maximum number of bytes per tensor (limits how many copies
        of the tensor we cache).
    """

    def __init__(self, memory_limit: int):
        self.memory_limit = memory_limit

    def dim_is_summarized(self, size: int):
        return size > 10 and size != 31


def get_sum_abs_stats(
    x: Tensor, dim: int, stats_type: str
) -> Tuple[Tensor, int]:
    """Returns the sum-of-absolute-value of this Tensor, for each index into
    the specified axis/dim of the tensor.

    Args:
      x:
        Tensor, tensor to be analyzed
      dim:
        Dimension with 0 <= dim < x.ndim
      stats_type:
        Either "mean-abs" in which case the stats represent the mean absolute
        value, or "pos-ratio" in which case the stats represent the proportion
        of positive values (actually: the tensor is count of positive values,
        count is the count of all values).

    Returns:
      (sum_abs, count) where sum_abs is a Tensor of shape (x.shape[dim],),
      and the count is an integer saying how many items were counted in
      each element of sum_abs.
    """
    if stats_type == "mean-abs":
        x = x.abs()
    else:
        assert stats_type == "pos-ratio"
        x = (x > 0).to(dtype=torch.float)

    orig_numel = x.numel()
    sum_dims = [d for d in range(x.ndim) if d != dim]
    x = torch.sum(x, dim=sum_dims)
    count = orig_numel // x.numel()
    x = x.flatten()

    return x, count


def get_diagnostics_for_dim(
    dim: int,
    tensors: List[Tensor],
    options: TensorDiagnosticOptions,
    sizes_same: bool,
    stats_type: str,
) -> str:
    """This function gets diagnostics for a dimension of a module.

    Args:
      dim:
        The dimension to analyze, with 0 <= dim < tensors[0].ndim
      tensors:
        List of cached tensors to get the stats
      options:
        Options object
      sizes_same:
        True if all the tensor sizes are the same on this dimension
        stats_type: either "mean-abs" or "pos-ratio", dictates the type of
        stats we accumulate, mean-abs is mean absolute value, "pos-ratio" is
        proportion of positive to nonnegative values.

    Returns:
      Diagnostic as a string, either percentiles or the actual values,
      see the code.
    """

    # stats_and_counts is a list of pair (Tensor, int)
    stats_and_counts = [get_sum_abs_stats(x, dim, stats_type) for x in tensors]
    stats = [x[0] for x in stats_and_counts]
    counts = [x[1] for x in stats_and_counts]
    if sizes_same:
        stats = torch.stack(stats).sum(dim=0)
        count = sum(counts)
        stats = stats / count
    else:
        stats = [x[0] / x[1] for x in stats_and_counts]
        stats = torch.cat(stats, dim=0)

    # If `summarize` we print percentiles of the stats;
    # else, we print out individual elements.
    summarize = (not sizes_same) or options.dim_is_summarized(stats.numel())
    if summarize:
        # Print out percentiles.
        stats = stats.sort()[0]
        num_percentiles = 10
        size = stats.numel()
        percentiles = []
        for i in range(num_percentiles + 1):
            index = (i * (size - 1)) // num_percentiles
            percentiles.append(stats[index].item())
        percentiles = ["%.2g" % x for x in percentiles]
        percentiles = " ".join(percentiles)
        return f"percentiles: [{percentiles}]"
    else:
        stats = stats.tolist()
        stats = ["%.2g" % x for x in stats]
        stats = "[" + " ".join(stats) + "]"
        return stats


def print_diagnostics_for_dim(
    name: str, dim: int, tensors: List[Tensor], options: TensorDiagnosticOptions
):
    """This function prints diagnostics for a dimension of a tensor.

    Args:
      name:
        The tensor name.
      dim:
        The dimension to analyze, with 0 <= dim < tensors[0].ndim.
      tensors:
        List of cached tensors to get the stats.
      options:
        Options object.
    """

    for stats_type in ["mean-abs", "pos-ratio"]:
        # stats_type will be "mean-abs" or "pos-ratio".
        sizes = [x.shape[dim] for x in tensors]
        sizes_same = all([x == sizes[0] for x in sizes])
        s = get_diagnostics_for_dim(
            dim, tensors, options, sizes_same, stats_type
        )

        min_size = min(sizes)
        max_size = max(sizes)
        size_str = f"{min_size}" if sizes_same else f"{min_size}..{max_size}"
        print(f"module={name}, dim={dim}, size={size_str}, {stats_type} {s}")


class TensorDiagnostic(object):
    """This class is not directly used by the user, it is responsible for
    collecting diagnostics for a single parameter tensor of a torch.nn.Module.

    Args:
      opts:
        Options object.
      name:
        The tensor name.
    """

    def __init__(self, opts: TensorDiagnosticOptions, name: str):
        self.name = name
        self.opts = opts
        # A list to cache the tensors.
        self.saved_tensors = []

    def accumulate(self, x):
        """Accumulate tensors."""
        if isinstance(x, Tuple):
            x = x[0]
        if not isinstance(x, Tensor):
            return
        if x.device == torch.device("cpu"):
            x = x.detach().clone()
        else:
            x = x.detach().to("cpu", non_blocking=True)
        self.saved_tensors.append(x)
        num = len(self.saved_tensors)
        if num & (num - 1) == 0:  # power of 2..
            self._limit_memory()

    def _limit_memory(self):
        """Only keep the newly cached tensors to limit memory."""
        if len(self.saved_tensors) > 1024:
            self.saved_tensors = self.saved_tensors[-1024:]
            return

        tot_mem = 0.0
        for i in reversed(range(len(self.saved_tensors))):
            tot_mem += (
                self.saved_tensors[i].numel()
                * self.saved_tensors[i].element_size()
            )
            if tot_mem > self.opts.memory_limit:
                self.saved_tensors = self.saved_tensors[i:]
                return

    def print_diagnostics(self):
        """Print diagnostics for each dimension of the tensor."""
        if len(self.saved_tensors) == 0:
            print("{name}: no stats".format(name=self.name))
            return

        if self.saved_tensors[0].ndim == 0:
            # Ensure there is at least one dim.
            self.saved_tensors = [x.unsqueeze(0) for x in self.saved_tensors]

        ndim = self.saved_tensors[0].ndim
        for dim in range(ndim):
            print_diagnostics_for_dim(
                self.name, dim, self.saved_tensors, self.opts
            )


class ModelDiagnostic(object):
    """This class stores diagnostics for all tensors in the torch.nn.Module.

    Args:
      opts:
        Options object.
    """

    def __init__(self, opts: TensorDiagnosticOptions):
        # In this dictionary, the keys are tensors names and the values
        # are corresponding TensorDiagnostic objects.
        self.diagnostics = dict()
        self.opts = opts

    def __getitem__(self, name: str):
        if name not in self.diagnostics:
            self.diagnostics[name] = TensorDiagnostic(self.opts, name)
        return self.diagnostics[name]

    def print_diagnostics(self):
        """Print diagnostics for each tensor."""
        for k in sorted(self.diagnostics.keys()):
            self.diagnostics[k].print_diagnostics()


def attach_diagnostics(
    model: nn.Module, opts: TensorDiagnosticOptions
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
        # (matters for name, since the variable gets overwritten).
        # These closures don't really capture by value, only by
        # "the final value the variable got in the function" :-(
        def forward_hook(
            _module, _input, _output, _model_diagnostic=ans, _name=name
        ):
            if isinstance(_output, Tensor):
                _model_diagnostic[f"{_name}.output"].accumulate(_output)
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    _model_diagnostic[f"{_name}.output[{i}]"].accumulate(o)

        def backward_hook(
            _module, _input, _output, _model_diagnostic=ans, _name=name
        ):
            if isinstance(_output, Tensor):
                _model_diagnostic[f"{_name}.grad"].accumulate(_output)
            elif isinstance(_output, tuple):
                for i, o in enumerate(_output):
                    _model_diagnostic[f"{_name}.grad[{i}]"].accumulate(o)

        module.register_forward_hook(forward_hook)
        module.register_backward_hook(backward_hook)

    for name, parameter in model.named_parameters():

        def param_backward_hook(
            grad, _parameter=parameter, _model_diagnostic=ans, _name=name
        ):
            _model_diagnostic[f"{_name}.param_value"].accumulate(_parameter)
            _model_diagnostic[f"{_name}.param_grad"].accumulate(grad)

        parameter.register_hook(param_backward_hook)

    return ans


def _test_tensor_diagnostic():
    opts = TensorDiagnosticOptions(2 ** 20)

    diagnostic = TensorDiagnostic(opts, "foo")

    for _ in range(10):
        diagnostic.accumulate(torch.randn(50, 100) * 10.0)

    diagnostic.print_diagnostics()

    model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 80))

    diagnostic = attach_diagnostics(model, opts)
    for _ in range(10):
        T = random.randint(200, 300)
        x = torch.randn(T, 100)
        y = model(x)
        y.sum().backward()

    diagnostic.print_diagnostics()


if __name__ == "__main__":
    _test_tensor_diagnostic()
