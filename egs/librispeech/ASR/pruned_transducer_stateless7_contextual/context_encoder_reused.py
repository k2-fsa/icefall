import torch
from context_encoder import ContextEncoder
import copy

class ContextEncoderReused(ContextEncoder):
    def __init__(
        self,
        decoder,
        decoder_dim: int = None,
        output_dim: int = None,
        num_lstm_layers: int = None,
        num_lstm_directions: int = None,
        drop_out: float = 0.1,
    ):
        super(ContextEncoderReused, self).__init__()
        # self.num_lstm_layers = num_lstm_layers
        # self.num_lstm_directions = num_lstm_directions
        # self.decoder_dim = decoder_dim

        hidden_size = output_dim * 2  # decoder_dim
        self.rnn = torch.nn.LSTM(
            input_size=decoder_dim, 
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True, 
            bidirectional=(num_lstm_directions == 2), 
            dropout=0.1 if num_lstm_layers > 1 else 0
        )
        self.linear = torch.nn.Linear(
            hidden_size * num_lstm_directions, 
            output_dim
        )
        self.drop_out = torch.nn.Dropout(drop_out)

        self.decoder = decoder

        self.bi_encoders = False

        # TODO: Do we need some relu layer?
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        word_list, 
        word_lengths,
        is_encoder_side=None,
    ):
        sos_id = self.decoder.blank_id
        sos_list = torch.full((word_list.shape[0], 1), sos_id).to(word_list.device)
        sos_word_list = torch.cat((sos_list, word_list), 1)
        word_lengths = [x + 1 for x in word_lengths]

        # sos_word_list: (N, U)
        # decoder_out:   (N, U, decoder_dim)
        out = self.decoder(sos_word_list)

        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, 
            batch_first=True, 
            lengths=word_lengths, 
            enforce_sorted=False
        )
        output, (hn, cn) = self.rnn(out)  # use default all zeros (h_0, c_0)

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
        final_h = self.linear(final_h)
        # final_h = drop_out(final_h)

        return final_h
