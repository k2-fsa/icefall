import torch
from context_encoder import ContextEncoder
import torch.nn.functional as F


class ContextEncoderPretrained(ContextEncoder):
    def __init__(
        self,
        vocab_size: int = None,
        context_encoder_dim: int = None,
        output_dim: int = None,
        num_layers: int = None,
        num_directions: int = None,
        drop_out: float = 0.3,
    ):
        super(ContextEncoderPretrained, self).__init__()

        self.drop_out = torch.nn.Dropout(drop_out)
        self.linear1 = torch.nn.Linear(
            context_encoder_dim,  # 768
            256,
        )
        self.linear3 = torch.nn.Linear(
            256,
            256,
        )
        self.linear4 = torch.nn.Linear(
            256,
            256,
        )
        self.linear2 = torch.nn.Linear(
            256,
            output_dim
        )
        self.sigmoid = torch.nn.Sigmoid()

        self.bi_encoders = False

    def forward(
        self, 
        word_list, 
        word_lengths,
        is_encoder_side=None,
    ):
        out = word_list  # Shape: N*L*D
        # out = self.drop_out(out)
        out = self.sigmoid(self.linear1(out))  # Note: ReLU may not be a good choice here
        out = self.sigmoid(self.linear3(out))
        out = self.sigmoid(self.linear4(out))
        # out = self.drop_out(out)
        out = self.linear2(out)
        return out
