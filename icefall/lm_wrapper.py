# Copyright (c)  2022  Xiaomi Corporation (authors: Xiaoyu Yang)
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

import argparse
import logging

import torch

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.rnn_lm.model import RnnLmModel
from icefall.transformer_lm.model import TransformerLM
from icefall.utils import AttributeDict, str2bool


class LmScorer(torch.nn.Module):
    """This is a wrapper for NN LMs
    The language models supported include:
        RNN,
        Transformer
    """

    def __init__(
        self,
        lm_type: str,
        params: AttributeDict,
        device,
        lm_scale: float = 0.3,
    ):
        super(LmScorer, self).__init__()
        assert lm_type in ["rnn", "transformer"], f"{lm_type} is not supported"
        self.lm_type = lm_type
        self.lm = self.get_lm(lm_type, device, params)
        self.lm_scale = lm_scale
        self.params = params

    @classmethod
    def add_arguments(cls, parser):
        # LM general arguments
        parser.add_argument(
            "--lm-vocab-size",
            type=int,
            default=500,
        )

        parser.add_argument(
            "--lm-epoch",
            type=int,
            default=7,
            help="""Which epoch to be used
            """,
        )

        parser.add_argument(
            "--lm-avg",
            type=int,
            default=1,
            help="""Number of checkpoints to be averaged
            """,
        )

        parser.add_argument("--lm-exp-dir", type=str, help="Path to LM experiments")

        # Now RNNLM related arguments
        parser.add_argument(
            "--rnn-lm-embedding-dim",
            type=int,
            default=2048,
            help="Embedding dim of the model",
        )

        parser.add_argument(
            "--rnn-lm-hidden-dim",
            type=int,
            default=2048,
            help="Hidden dim of the model",
        )

        parser.add_argument(
            "--rnn-lm-num-layers",
            type=int,
            default=3,
            help="Number of RNN layers the model",
        )

        parser.add_argument(
            "--rnn-lm-tie-weights",
            type=str2bool,
            default=True,
            help="""True to share the weights between the input embedding layer and the
            last output linear layer
            """,
        )

        # Now transformers
        parser.add_argument(
            "--transformer-lm-exp-dir", type=str, help="Directory of transformer LM exp"
        )

        parser.add_argument(
            "--transformer-lm-dim-feedforward",
            type=int,
            default=2048,
            help="Dimension of FFW module in transformer",
        )

        parser.add_argument(
            "--transformer-lm-encoder-dim",
            type=int,
            default=768,
            help="Encoder dimension of transformer",
        )

        parser.add_argument(
            "--transformer-lm-embedding-dim",
            type=int,
            default=768,
            help="Input embedding dimension of transformer",
        )

        parser.add_argument(
            "--transformer-lm-nhead",
            type=int,
            default=8,
            help="Number of attention heads in transformer",
        )

        parser.add_argument(
            "--transformer-lm-num-layers",
            type=int,
            default=16,
            help="Number of encoder layers in transformer",
        )

        parser.add_argument(
            "--transformer-lm-tie-weights",
            type=str2bool,
            default=True,
            help="If tie weights in transformer LM",
        )

    def get_lm(self, lm_type: str, device, params: AttributeDict) -> torch.nn.Module:
        """Return the neural network LM

        Args:
            lm_type (str): Type name of NN LM
        """
        if lm_type == "rnn":
            model = RnnLmModel(
                vocab_size=params.lm_vocab_size,
                embedding_dim=params.rnn_lm_embedding_dim,
                hidden_dim=params.rnn_lm_hidden_dim,
                num_layers=params.rnn_lm_num_layers,
                tie_weights=params.rnn_lm_tie_weights,
            )

            if params.lm_avg == 1:
                load_checkpoint(
                    f"{params.lm_exp_dir}/epoch-{params.lm_epoch}.pt", model
                )
                model.to(device)
            else:
                start = params.lm_epoch - params.lm_avg + 1
                filenames = []
                for i in range(start, params.lm_epoch + 1):
                    if start >= 0:
                        filenames.append(f"{params.lm_exp_dir}/epoch-{i}.pt")
                logging.info(f"averaging {filenames}")
                model.to(device)
                model.load_state_dict(average_checkpoints(filenames, device=device))

        elif lm_type == "transformer":
            model = TransformerLM(
                vocab_size=params.lm_vocab_size,
                d_model=params.transformer_lm_encoder_dim,
                embedding_dim=params.transformer_lm_embedding_dim,
                dim_feedforward=params.transformer_lm_dim_feedforward,
                nhead=params.transformer_lm_nhead,
                num_layers=params.transformer_lm_num_layers,
                tie_weights=params.transformer_lm_tie_weights,
                params=params,
            )

            if params.lm_avg == 1:
                load_checkpoint(
                    f"{params.lm_exp_dir}/epoch-{params.lm_epoch}.pt", model
                )
                model.to(device)
            else:
                start = params.lm_epoch - params.lm_avg + 1
                filenames = []
                for i in range(start, params.lm_epoch + 1):
                    if start >= 0:
                        filenames.append(f"{params.lm_exp_dir}/epoch-{i}.pt")
                logging.info(f"averaging {filenames}")
                model.to(device)
                model.load_state_dict(average_checkpoints(filenames, device=device))
        else:
            raise NotImplementedError()

        return model

    def score_token(self, x: torch.Tensor, x_lens: torch.Tensor, state=None):
        """Score the input and return the prediction
        This requires the lm to have the method `score_token`
        Args:
            x (torch.Tensor): Input tokens
            x_lens (torch.Tensor): Length of the input tokens
            state (optional): LM states

        """
        return self.lm.score_token(x, x_lens, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    LmScorer.add_arguments(parser)
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    Scorer = LmScorer(params=params, device=device)
    Scorer.eval()

    x = (
        torch.tensor([[1, 4, 19, 256, 77], [1, 4, 19, 256, 77]])
        .to(device)
        .to(torch.int64)
    )
    x_lens = torch.tensor([5, 5]).to(device)

    state = None

    score, state = Scorer.score(x, x_lens)
    print(score.shape)
    print(score[0])
    print(score[1])
