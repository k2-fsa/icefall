import math
import torch.nn.functional as F


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz 
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2 

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder
~                                                                        
