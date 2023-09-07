# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

import torch
from kaldi_hmm_gmm import DecodableInterface


class CtcDecodable(DecodableInterface):
    """This class implements the interface
    https://github.com/kaldi-asr/kaldi/blob/master/src/itf/decodable-itf.h
    """

    def __init__(self, nnet_output: torch.Tensor):
        DecodableInterface.__init__(self)
        assert nnet_output.ndim == 2, nnet_output.shape
        self.nnet_output = nnet_output

    def log_likelihood(self, frame: int, index: int) -> float:
        # Note: We need to use index - 1 here since
        # all the input labels of the H are incremented during graph
        # construction
        return self.nnet_output[frame][index - 1].item()

    def is_last_frame(self, frame: int) -> bool:
        return frame == self.nnet_output.shape[0] - 1

    def num_frames_ready(self) -> int:
        return self.nnet_output.shape[0]

    def num_indices(self) -> int:
        return self.nnet_output.shape[1]
