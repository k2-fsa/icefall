# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import Optional, Tuple, List

import k2
import torch
import logging
import torch.nn as nn
from torch import Tensor
from scaling import ScaledLinear, convert_num_channels, SwashR
import math
from icefall.utils import make_pad_mask, time_warp



class DenoisingAsrModel(nn.Module):
    def __init__(
            self,
            #speech_embed: nn.Module,
            encoder: nn.Module,
            encoder_dim: int,
            text_embed_dim: int,
            vocab_size: int,
            time_embed_dim: int,
    ):
        """
        TODO
        """
        super().__init__()

        self.speech_scale = 0.5
        self.encoder = encoder
        self.encoder_dim = encoder_dim

        # s is the time value for the speech, 0 <= s <= 1.
        # t is the time value for the symbols, 0 <= t <= 1.
        self.time_embed_dim = time_embed_dim
        self.st_embed = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # randomly initialize text embedding and do not train it.
        text_embed_scale = 0.25  # this will ensure that later steps still "matter".
        self.text_embed = FixedEmbedding(vocab_size, text_embed_dim, scale=text_embed_scale)

        self.text_in_proj = nn.Linear(text_embed_dim, encoder_dim)
        self.text_out_proj = nn.Linear(encoder_dim, text_embed_dim)

        # for now just hardcode
        speech_channels = 80
        speech_subsample = 4
        self.speech_out_proj = nn.Linear(encoder_dim,
                                         speech_channels * speech_subsample)

        self.speech_in_proj = nn.Linear(speech_channels * speech_subsample,
                                        encoder_dim)



    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A Tensor of dtype long, indexed [utt][symbol], padded with symbol 0
            on the right.  There is no BOS or EOS symbol.

        Returns:
          Returns flow-matching loss values for symbols and speech respectively.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        batch_size = x.shape[0]
        assert x.shape[0] == x_lens.shape[0] == y.shape[0], (x.shape, x_lens.shape, y.shape)

        s = torch.rand(batch_size, device=x.device)  # time-value for speech.
        t = torch.rand(batch_size, device=x.device)  # time-value for text.


        st = self.st_embed(torch.cat((timestep_embedding(s, self.time_embed_dim),
                                      timestep_embedding(t, self.time_embed_dim)), dim=1))
        # st: (batch_size, time_embed_dim)

        (batch_size, speech_seq_len, num_freqs) = x.shape

        device = x.device
        x1 = x * self.speech_scale  # scale log-mels by 0.1 to be better matched to normal distribution.
        x0 = torch.randn_like(x1)
        xs = (x1 * s[:, None, None]) + (x0 * (1 - s[:, None, None]))
        # x1, x0, xs: (batch_size, seq_len, 80)
        xV = x1 - x0  # xV means x velocity.  (batch_size, speech_seq_len, 80)

        padding = (4 - (speech_seq_len % 4)) % 4
        xs = torch.nn.functional.pad(xs, (0, 0, 0, padding))
        xs = xs.reshape(batch_size, -1, 4 * num_freqs)
        xs_embed = self.speech_in_proj(xs)
        x_lens_embed = x_lens // 4

        xs_embed = xs_embed.permute(1, 0, 2)  # (embed_seq_len, batch_size, encoder_dim)
        embed_seq_len = xs_embed.shape[0]

        with torch.amp.autocast('cuda', enabled=False):
            y = randomly_pad_to_lengths(y, y_lens, torch.minimum(x_lens_embed, y_lens + y_lens // 4), embed_seq_len)
        # now y: (batch_size, seq_len)
        y1 = self.text_embed(y)
        # now y1: (batch_size, seq_len, text_embed_dim)
        y0 = torch.randn_like(y1)
        yt = (y1 * t[:, None, None]) + (y0 * (1 - t[:, None, None]))
        # yt: (batch_size, seq_len, text_embed_dim)
        yt_embed = self.text_in_proj(yt).permute(1, 0, 2)  # (embed_seq_len, batch_size, encoder_dim)
        yV = y1 - y0  # yV means y velocity.  (batch_size, embed_seq_len, text_embed_dim)

        encoder_in = xs_embed + yt_embed

        src_key_padding_mask = torch.arange(0, embed_seq_len, device=x.device) >= x_lens_embed.unsqueeze(-1)  # (batch-size, max_x_len)

        encoder_out = self.encoder(encoder_in, st, x_lens_embed, src_key_padding_mask)
        (embed_seq_len, batch_size, _encoder_dim) = encoder_out.shape

        xU = self.speech_out_proj(encoder_out)
        xU = xU.permute(1, 0, 2).reshape(batch_size, embed_seq_len * 4, -1)
        xU = xU[:, :speech_seq_len]  # (batch_size, speech_seq_len, 80)

        # don't use x_mask in training, this will simplify inference.
        # x_mask = (torch.arange(0, speech_seq_len, device=x.device) < x_lens.unsqueeze(-1)).unsqueeze(-1)
        # x_mask: # (batch-size, speech_seq_len, 1).

        x_loss = ((xV - xU) ** 2).mean(dim=-1).sum()

        yU = self.text_out_proj(encoder_out)
        yU = yU.permute(1, 0, 2)  # (batch_size, embed_seq_len, text_embed_dim)

        #y_mask = torch.logical_not(src_key_padding_mask).unsqueeze(-1)
        y_loss = ((yV - yU) ** 2).mean(dim=-1).sum()

        return x_loss, y_loss  # speech_loss, text_loss


    def infer(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        eps: float,
        num_steps: int,
    ) -> List[List[int]]:
        """
        Does inference.  Starting from random noise representing the text, does inference
        for a number of steps and then converts the text representation to integers.
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          eps:
            The 't' value to start inference from, e.g. 1.0e-04
          num_steps:
            The number of inference steps to use.

        Returns:
          Returns the inference result as a list of lists of symbols, with blanks (symbol zero)
          removed.
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        batch_size = x.shape[0]
        assert x.shape[0] == x_lens.shape[0]

        s = torch.ones(batch_size, device=x.device)  # time-value for speech is 1.0 throughout.
        xs = x * self.speech_scale  # scale log-mels by 0.1 to be better matched to normal distribution.
        # xs is the same as x1, because s == 1.0, in inference there is no noise on the speech.
        (batch_size, speech_seq_len, num_freqs) = xs.shape
        padding = (4 - (speech_seq_len % 4)) % 4
        xs = torch.nn.functional.pad(xs, (0, 0, 0, padding))
        xs = xs.reshape(batch_size, -1, 4 * num_freqs)
        xs_embed = self.speech_in_proj(xs)
        x_lens_embed = x_lens // 4
        xs_embed = xs_embed.permute(1, 0, 2)  # (embed_seq_len, batch_size, encoder_dim)
        (embed_seq_len, batch_size, encoder_dim) = xs_embed.shape
        src_key_padding_mask = torch.arange(0, embed_seq_len, device=x.device) >= x_lens_embed.unsqueeze(-1)  # (batch-size, max_x_len)
        text_embed_dim = self.text_embed.weight.shape[1]

        delta_t = (1.0 - eps) / num_steps

        yt = torch.randn(embed_seq_len, batch_size, text_embed_dim, device=x.device)  # start with noise at t ~ 0

        for step in range(num_steps):
            t = torch.full((batch_size,), eps + step * delta_t, device=x.device)  # time-value for text.
            st = self.st_embed(torch.cat((timestep_embedding(s, self.time_embed_dim),
                                          timestep_embedding(t, self.time_embed_dim)), dim=1))
            # st: (batch_size, time_embed_dim)


            yt_embed = self.text_in_proj(yt)  # (embed_seq_len, batch_size, encoder_dim)
            encoder_in = xs_embed + yt_embed
            encoder_out = self.encoder(encoder_in, st, x_lens_embed, src_key_padding_mask)
            yU = self.text_out_proj(encoder_out)

            yt = yt + yU * delta_t


        yt = yt.permute(1, 0, 2)  # (batch_size, seq_len, text_embed_dim)
        tokens, residual = find_closest_tokens(yt, self.text_embed.weight)

        logging.info(f"Avg residual is {residual}")

        tokens = tokens.tolist()
        # remove blanks.
        tokens = [ [ s for s in sent if s != 0 ] for sent in tokens  ]

        return tokens



class FixedEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.register_buffer('weight', scale * torch.randn(vocab_size, embed_dim),
                             persistent=True)

    def forward(self, y: Tensor):
        y_shape = y.shape
        ans = torch.index_select(self.weight, 0, y.flatten())
        return ans.reshape(*y_shape, -1)



def find_closest_tokens(y: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Find closest token indexes to embedding vectors.
    Args:
          y: (..., embed_dim), the embeddings to match to weights.
    weights: (num_tokens, embed_dim), the embedding vectors for each token.

    Returns:  (tokens, avg_residual)
         tokens: (...), a LongTensor containing the indexes of the closest tokens
      avg_residual: a LongTensor containing the average difference (rms of elements)
               between embeddings and weights.
    """
    yy = (y ** 2).sum(dim=-1)  # (...)
    ww = (weights ** 2).sum(dim=-1)  # (num_tokens,)
    yw = torch.matmul(y, weights.t()) # (..., num_tokens)
    # (y - w) ** 2 = y**2 + w**2 - 2 yw

    residuals = yy.unsqueeze(-1) + ww - 2 * yw
    residuals, tokens = torch.min(residuals, dim=-1)

    embed_dim = weights.shape[1]
    return tokens, (residuals.mean() / embed_dim).sqrt()



def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def randomly_pad_to_lengths(y: Tensor,
                            y_lens: Tensor,
                            x_lens: Tensor,
                            max_x_len: int):
    """
    Randomly insert blanks (symbol 0) into the symbol-sequences in y, with lengths y_lens, so that
    they have lengths x_lens.  All tensor are LongTensors (dtype torch.long)
      Args:
            y:  (batch_size, max_y_len): the symbols; all positions less than the corresponding y_lens value
                    are expected to be nonzero.
        y_lens: the lengths of the sequences in y, we expect that 1 <= y_lens <= max_y_len
        x_lens: the lengths of the sequences we want to pad to, we expect that y_lens <= x_lens <= max_x_len.
    """
    # checking that each y is not longer than corresponding x.
    debug = True #(__name__ == '__main__')
    length_diff = x_lens - y_lens
    if debug:
        assert length_diff.min() >= 0

    (batch_size, max_y_len) = y.shape


    y_mask = torch.arange(0, max_y_len + 1, device=y.device) >= y_lens.unsqueeze(-1)  # (batch-size, max_y_len)
    # y_mask is True for masked, i.e. non-valid, positions

    # cut_points are points at which we divide up the interval [0..y_len-x_len] which is
    # the amount by which we want to pad.  We want to get y_len + 1 "padding lengths" that
    # sum to y_len-x_len.  We get these by taking the numbers: [ 0, <random numbers of count y_len>, 1 , 1... ],
    # multiplying by (y_len-x_len), so we have: [ 0, <random numbers of count y_len>, y_len-x_len, y_len-x_len.. ],
    # and take the differences between each one and the next, so we get:
    # [ <random padding counts of count y_len+1>, 0, 0, ... ] and the counts add up to y_len-x_len.
    #
    cut_points = torch.rand(batch_size, max_y_len + 2, device=y.device)
    cut_points[:, 1:].masked_fill_(y_mask, 1.0)
    cut_points[:, 0] = 0.0
    cut_points = cut_points * length_diff.unsqueeze(-1)
    cut_points = cut_points.sort(dim=1)[0]
    cut_points = cut_points.round().to(torch.long)
    num_pad = cut_points[:, 1:] - cut_points[:, :-1]



    num_symbols = torch.empty(batch_size, 2 * max_y_len, device=y.device, dtype=torch.long)
    num_symbols[:, 1::2] = (1 - y_mask[:, :-1].to(torch.long))  # the actual symbols have length 1.
    num_symbols[:, 0:-1:2] = num_pad[:, :-1]  # assign the number of padding symbols for each position.
    # we don't need the last padding length,  it doesn't determine any symbol position.

    symbol_positions = num_symbols.cumsum(dim=1)
    symbol_positions = symbol_positions[:, 0::2]

    # the "+ 1" is because the symbol_positions will actually contain, in the padding
    # positions, a number equal to the corresponding values in x_lens; and this may
    # be out of range in the scatter_ unless we add one padding element.
    padded_symbols = torch.zeros(batch_size, max_x_len + 1, device=y.device, dtype=torch.long)
    padded_symbols.scatter_(dim=1, index=symbol_positions, src=y)
    padded_symbols = padded_symbols[:, :-1]  # remove the one padding position
    x_mask = torch.arange(0, max_x_len, device=y_lens.device) < x_lens.unsqueeze(-1)
    if debug:
        assert torch.all(padded_symbols == padded_symbols * x_mask)
    return padded_symbols


def _test_find_closest_tokens():
    vocab_size = 10
    embed_dim = 30
    text_embed = FixedEmbedding(vocab_size, embed_dim)
    tokens = torch.randint(0, vocab_size, (3, 4), dtype=torch.long)

    embeddings = text_embed(tokens)
    embeddings = embeddings + 0.05 * torch.randn_like(embeddings)

    tokens2, residual = find_closest_tokens(embeddings, text_embed.weight)
    print("Residual = ", residual) # should be around 0.05.
    assert torch.all(tokens2 == tokens)


def _test_randomly_distribute_labels():
    y = torch.tensor([  [ 1, 2, 3, 4 ],  [ 5, 6, 7, 0 ], [ 8, 9, 0, 0 ] ])
    y_lens = torch.tensor([ 4, 3, 2 ] )
    x_lens = torch.tensor([ 8, 6, 5 ])
    max_x_len = 7
    y = randomly_pad_to_lengths(y, y_lens, x_lens, max_x_len)
    print("y_padded = ", y)




if __name__ == '__main__':
    _test_find_closest_tokens()
    for _ in range(10):
        _test_randomly_distribute_labels()
