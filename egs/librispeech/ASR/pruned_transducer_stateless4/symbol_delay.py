#!/usr/bin/env python3
#
# Copyright 2022 Xiaomi Corporation (Author: Wei Kang)
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
"""
Usage:
./pruned_transducer_stateless4/symbol_delay.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless4/exp \
    --max-duration 600
"""


import argparse
import logging
from pathlib import Path
from typing import Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule

from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    add_sos,
    setup_logger,
    str2bool,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless4/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--simulate-streaming",
        type=str2bool,
        default=False,
        help="""Whether to simulate streaming in decoding, this is a good way to
        test a streaming model.
        """,
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=16,
        help="The chunk size for decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--left-context",
        type=int,
        default=64,
        help="""left context can be seen during decoding
        (in frames after subsampling)
        """,
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)"
        "part.",
    )

    parser.add_argument(
        "--delay-penalty-simple",
        type=float,
        default=0.0,
        help="""A constant value to penalize symbol delay, this may be
         needed when training with time masking, to avoid the time masking
         encouraging the network to delay symbols.
         """,
    )

    add_model_arguments(parser)

    return parser


def calculate_symbol_delay(
    params: AttributeDict,
    model: nn.Module,
    x: torch.Tensor,
    x_lens: torch.Tensor,
    y: k2.RaggedTensor,
    am_scale: float = 0.0,
    lm_scale: float = 0.0,
    delay_penalty_simple: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      x:
        A 3-D tensor of shape (N, T, C).
      x_lens:
        A 1-D tensor of shape (N,). It contains the number of frames in `x`
        before padding.
      y:
        A ragged tensor with 2 axes [utt][label]. It contains labels of each
        utterance.
      am_scale:
        The scale to smooth the loss with am (output of encoder network)
        part
      lm_scale:
        The scale to smooth the loss with lm (output of predictor network)
        part
      delay_penalty_simple:
        A constant value to penalize symbol delay.
    Returns:
      Return the symbol delay and the number of symbols.
    """
    assert x.ndim == 3, x.shape
    assert x_lens.ndim == 1, x_lens.shape
    assert y.num_axes == 2, y.num_axes

    assert x.size(0) == x_lens.size(0) == y.dim0

    if params.simulate_streaming:
        encoder_out, x_lens, _ = model.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            chunk_size=params.decode_chunk_size,
            left_context=params.left_context,
            simulate_streaming=True,
        )
    else:
        encoder_out, x_lens = model.encoder(
            x=x, x_lens=x_lens
        )
    assert torch.all(x_lens > 0)

    # Now for the decoder, i.e., the prediction network
    row_splits = y.shape.row_splits(1)
    y_lens = row_splits[1:] - row_splits[:-1]

    blank_id = model.decoder.blank_id
    sos_y = add_sos(y, sos_id=blank_id)

    # sos_y_padded: [B, S + 1], start with SOS.
    sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

    # decoder_out: [B, S + 1, decoder_dim]
    decoder_out = model.decoder(sos_y_padded)

    # Note: y does not start with SOS
    # y_padded : [B, S]
    y_padded = y.pad(mode="constant", padding_value=0)

    y_padded = y_padded.to(torch.int64)
    boundary = torch.zeros(
        (x.size(0), 4), dtype=torch.int64, device=x.device
    )
    boundary[:, 2] = y_lens
    boundary[:, 3] = x_lens

    lm = model.simple_lm_proj(decoder_out)
    am = model.simple_am_proj(encoder_out)

    simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
        lm=lm.float(),
        am=am.float(),
        symbols=y_padded,
        termination_symbol=blank_id,
        lm_only_scale=lm_scale,
        am_only_scale=am_scale,
        boundary=boundary,
        delay_penalty=delay_penalty_simple,
        return_grad=True,
    )

    B, S, T0 = px_grad.shape
    T = T0 - 1
    if boundary is None:
        offset = torch.tensor(
            (T - 1) / 2,
            dtype=px_grad.dtype,
            device=px_grad.device,
        ).expand(B, 1, 1)
        total_syms = S * B
    else:
        offset = (boundary[:, 3] - 1) / 2
        total_syms = torch.sum(boundary[:, 2])
    offset = torch.arange(
        T0, device=px_grad.device
    ).reshape(1, 1, T0) - offset.reshape(B, 1, 1)
    sym_delay = px_grad * offset
    sym_delay = torch.sum(sym_delay)

    return (sym_delay, total_syms)


def calculate_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> MetricsTracker:
    """
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the symbol delay and number of symbols in a MetricsTracker object.
    """
    device = next(model.parameters()).device

    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    sym_delay, total_syms = calculate_symbol_delay(
        params=params,
        model=model,
        x=feature,
        x_lens=feature_lens,
        y=y,
        am_scale=params.am_scale,
        lm_scale=params.lm_scale,
        delay_penalty_simple=params.delay_penalty_simple
    )
    delay = MetricsTracker()
    delay["symbols"] = total_syms.detach().cpu().item()
    delay["sym_delay"] = sym_delay.detach().cpu().item()

    return delay


def calculate_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> MetricsTracker:
    """Decode dataset.
    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
    Returns:
      Return the symbol delay and number of symbols in a MetricsTracker object.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    tot_delay = MetricsTracker()
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]

        current_delay = calculate_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
        )

        tot_delay += current_delay

        num_cuts += len(texts)

        if batch_idx % 20 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(
                f"batch {batch_str}, cuts processed until now is {num_cuts}"
            )

    return tot_delay


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_delay: MetricsTracker,
):
    delay_path = (
        params.res_dir / f"symbol-delay-{test_set_name}-{params.suffix}.txt"
    )
    with open(delay_path, "w") as f:
        print(f"{results_delay}", file=f)

    logging.info(f"symbol delay for {test_set_name} is : {results_delay}")


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "symbol_delay"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.simulate_streaming:
        params.suffix += f"-streaming-chunk-size-{params.decode_chunk_size}"
        params.suffix += f"-left-context-{params.left_context}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/symbol-delay-{params.suffix}")

    logging.info("Start to calculate symbol delay.")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    if params.simulate_streaming:
        assert (
            params.causal_convolution
        ), "Decoding in streaming requires causal convolution"

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg + 1]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_other_dl = librispeech.test_dataloaders(test_other_cuts)

    test_sets = ["test-clean", "test-other"]
    test_dl = [test_clean_dl, test_other_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_delay = calculate_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_delay=results_delay,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
