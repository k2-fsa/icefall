import torch
from context_encoder import ContextEncoder
import copy

class ContextEncoderLSTM(ContextEncoder):
    def __init__(
        self,
        vocab_size: int = None,
        context_encoder_dim: int = None,
        output_dim: int = None,
        num_layers: int = None,
        num_directions: int = None,
        drop_out: float = 0.1,
        bi_encoders: bool = False,
    ):
        super(ContextEncoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.context_encoder_dim = context_encoder_dim

        torch.manual_seed(42)
        self.embed = torch.nn.Embedding(
            vocab_size, 
            context_encoder_dim
        )
        self.rnn = torch.nn.LSTM(
            input_size=context_encoder_dim, 
            hidden_size=context_encoder_dim, 
            num_layers=self.num_layers,
            batch_first=True, 
            bidirectional=(self.num_directions == 2), 
            dropout=0.0 if self.num_layers > 1 else 0
        )
        self.linear = torch.nn.Linear(
            context_encoder_dim * self.num_directions, 
            output_dim
        )
        self.drop_out = torch.nn.Dropout(drop_out)

        # TODO: Do we need some relu layer?
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

        self.bi_encoders = bi_encoders
        if bi_encoders:
            # Create the decoder/predictor side of the context encoder
            self.embed_dec = copy.deepcopy(self.embed)
            self.rnn_dec = copy.deepcopy(self.rnn)
            self.linear_dec = copy.deepcopy(self.linear)
            self.drop_out_dec = copy.deepcopy(self.drop_out)

    def forward(
        self, 
        word_list, 
        word_lengths,
        is_encoder_side=None,
    ):
        if is_encoder_side is None or is_encoder_side is True:
            embed = self.embed
            rnn = self.rnn
            linear = self.linear
            drop_out = self.drop_out
        else:
            embed = self.embed_dec
            rnn = self.rnn_dec
            linear = self.linear_dec
            drop_out = self.drop_out_dec

        out = embed(word_list)
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, 
            batch_first=True, 
            lengths=word_lengths, 
            enforce_sorted=False
        )
        output, (hn, cn) = rnn(out)  # use default all zeros (h_0, c_0)

        # # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/4
        # final_state = hn.view(
        #     self.num_layers, 
        #     self.num_directions,
        #     word_list.shape[0], 
        #     self.encoder_dim,
        # )[-1]  # Only the last layer
        # h_1, h_2 = final_state[0], final_state[1]
        # # X = h_1 + h_2                     # Add both states (needs different input size for first linear layer)
        # final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        # final_h = self.linear(final_h)

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer.
        # hidden[-2, :, : ] is the last of the forwards RNN 
        # hidden[-1, :, : ] is the last of the backwards RNN
        h_1, h_2 = hn[-2, :, : ] , hn[-1, :, : ]
        final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        final_h = linear(final_h)
        # final_h = drop_out(final_h)

        return final_h
