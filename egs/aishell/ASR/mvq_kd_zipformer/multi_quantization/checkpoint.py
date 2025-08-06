import torch
from torch import nn
from torch import Tensor
from typing import Tuple, Callable


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function: Callable, *args):
        # `function` must return either a Tensor or a tuple of Tensors
        ctx.function = function
        ctx.args = [x.detach() if isinstance(x, Tensor) else x
                    for x in args]
        for i in range(len(ctx.args)):
            if isinstance(args[i], Tensor) and args[i].requires_grad:
                ctx.args[i].requires_grad = True
        with torch.no_grad():
            ans = function(*args)

        return ans

    @staticmethod
    def backward(ctx, *ans_grads):
        if not any([ a is not None for a in ans_grads]):
            return [None] * len(ctx.args)
        with torch.enable_grad():
            ans = ctx.function(*ctx.args)
            if isinstance(ans, Tensor):
                assert len(ans_grads) == 1
                loss = (ans * ans_grads[0]).sum()
            else:
                assert len(ans_grads) == len(ans)
                loss = torch.stack([ (a * g).sum() for a, g in zip(ans, ans_grads)
                                     if g is not None ]).sum()

        loss.backward()
        return tuple([None] + [ a.grad if isinstance(a, Tensor) else None for a in ctx.args ])



def checkpoint(function, *args):
    return CheckpointFunction.apply(function, *args)




def _test1():
    x = torch.Tensor([0])
    y = torch.Tensor([1])
    y.requires_grad = True
    l = lambda x, y, trash: torch.stack((x, y))
    ans = checkpoint(l, x, y, None)
    #ans = l(x, y, None)
    print("ans = ", ans)
    (-ans).sum().backward()
    print("y grad = ", y.grad)


def _test2():
    x = torch.Tensor([0])
    y = torch.Tensor([1])
    x.requires_grad = True
    l = lambda x, y, trash: torch.stack((x, y))
    ans = checkpoint(l, x, y, None)
    ans = checkpoint(torch.sum, ans)
    #ans = l(x, y, None)
    print("ans = ", ans)
    (-ans).backward()
    print("x grad = ", x.grad)

if __name__ == '__main__':
    _test1()
    _test2()
