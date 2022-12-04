import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from icefall.utils import make_pad_mask


class FrameReducer(nn.Module):
    """The encoder output is first used to calculate
    the CTC posterior probability; then for each output frame,
    if its blank posterior is bigger than some thresholds,
    it will be simply discarded from the encoder output.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        ctc_output: torch.Tensor,
        blank_id: int,
    ) -> torch.Tensor:

        padding_mask = make_pad_mask(x_lens)
        non_blank_mask = (ctc_output[:, :, blank_id] < \
            math.log(0.9)) * (~padding_mask)
        T_range = torch.arange(x.shape[1], device=x.device)

        frames_list, lens_list = [], []
        for i in range(x.shape[0]):
            indexes = torch.masked_select(
                T_range,
                non_blank_mask[i],
            )
            frames = x[i][indexes]
            frames_list.append(frames)
            lens_list.append(frames.shape[0])
        x_fr = pad_sequence(frames_list).transpose(0, 1)
        x_lens_fr = torch.tensor(lens_list).to(device=x.device)

        return x_fr, x_lens_fr


if __name__ == "__main__":
    a = FrameReducer()
    x = torch.Tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
    x_lens = torch.tensor([2, 2, 2], dtype=int)
    ctc_output = torch.Tensor([[[1e10, 0, 0], [1, 0, 0]], [[0, 2, 3], [1e10, 2, 3]], [[0, 2, 3], [0, 2, 3]]])
    ctc_output = ctc_output.log_softmax(dim=2)

    x_fr, x_lens_fr = a(x, x_lens, ctc_output, 0)
    print(x==x_fr)  # True
    print(x_lens==x_lens_fr)  # True
