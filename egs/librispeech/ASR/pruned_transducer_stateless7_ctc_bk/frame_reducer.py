import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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
        ctc_posterior: torch.Tensor,
        blank_id: int,
    ) -> torch.Tensor:
        mask = ctc_posterior[:,:,blank_id] >= 0.9
        x_lens_fr = x_lens - mask.sum(axis=1)
        x_maxlen = x_lens_fr.max()
        mask = mask.unsqueeze(2).expand_as(x)
        x_fr_list = x[~mask].tolist()

        x_fr = []
        x_lens_cum = torch.cumsum(x_lens_fr, dim=0)
        frame_idx = 0
        one_frame_list = []
        for i in range(x_lens_cum[-1]):
            one_frame_list.append(
                x_fr_list[
                    i * x.size(2) : \
                    (i + 1) * x.size(2)
                ]
            )

            if i + 1 == x_lens_cum[frame_idx].item():
                x_fr.append(
                    one_frame_list + \
                    [[0.] * x.size(2)] * \
                    (x_maxlen - x_lens_fr[frame_idx].item())
                )

                frame_idx += 1
                one_frame_list = []

        x_fr = torch.tensor(x_fr, dtype=x.dtype).to(x.device)
        x_lens_fr = x_lens_fr.to(x_lens.device)

        return x_fr, x_lens_fr


if __name__ == "__main__":
    a = FrameReducer()
    x = torch.Tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
    x_lens = torch.tensor([2, 2, 2], dtype=int)
    ctc_posterior = torch.Tensor([[[0, 2, 3], [0, 2, 3]], [[0, 2, 3], [0, 2, 3]], [[0, 2, 3], [0, 2, 3]]])

    x_fr, x_lens_fr = a(x, x_lens, ctc_posterior, 0)
