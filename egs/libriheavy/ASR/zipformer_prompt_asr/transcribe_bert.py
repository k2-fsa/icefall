# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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

python ./zipformer_prompt_asr/transcribe_bert.py \
    --epoch 50 \
    --avg 10 \
    --exp-dir ./zipformer_prompt_asr/exp \
    --manifest-dir data/long_audios/long_audio.jsonl.gz \
    --pre-text-transform mixed-punc \
    --style-text-transform mixed-punc \
    --num-history 5 \
    --use-pre-text True \
    --use-gt-pre-text False


"""

import argparse
import logging
import math
import warnings
from pathlib import Path
from typing import List

import k2
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from decode_bert import _apply_style_transform
from lhotse import Fbank, load_manifest
from text_normalization import (
    lower_all_char,
    lower_only_alpha,
    ref_text_normalization,
    remove_non_alphabetic,
    train_text_normalization,
    upper_all_char,
    upper_only_alpha,
)
from tqdm import tqdm
from train_bert_encoder import (
    _encode_texts_as_bytes_with_tokenizer,
    add_model_arguments,
    get_params,
    get_tokenizer,
    get_transducer_model,
)

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
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
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="""Path to bpe.model.""",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/long_audios/long_audio.jsonl.gz",
        help="""This is the manfiest for long audio transcription.
        The cust are intended to be sorted, i.e first sort by recording ID and
        then sort by start timestamp""",
    )

    parser.add_argument(
        "--use-pre-text",
        type=str2bool,
        default=False,
        help="Whether use pre-text when decoding the current chunk",
    )

    parser.add_argument(
        "--use-style-prompt",
        type=str2bool,
        default=True,
        help="Use style prompt when evaluation",
    )

    parser.add_argument(
        "--pre-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc", "lower-punc"],
        default="mixed-punc",
        help="The style of content prompt, i.e pre_text",
    )

    parser.add_argument(
        "--style-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc", "lower-punc"],
        default="mixed-punc",
        help="The style of style prompt, i.e style_text",
    )

    parser.add_argument(
        "--num-history",
        type=int,
        default=2,
        help="How many previous chunks to look if using pre-text for decoding",
    )

    parser.add_argument(
        "--use-gt-pre-text",
        type=str2bool,
        default=False,
        help="Whether use gt pre text when using content prompt",
    )

    parser.add_argument(
        "--post-normalization",
        type=str2bool,
        default=True,
    )

    add_model_arguments(parser)

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()

    params.update(vars(args))

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    params.res_dir = params.exp_dir / "long_audio_transcribe"
    params.res_dir.mkdir(exist_ok=True)

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "beam_search" in params.method:
        params.suffix += f"-{params.method}-beam-size-{params.beam_size}"

    if params.use_pre_text:
        if params.use_gt_pre_text:
            params.suffix += f"-use-gt-pre-text-{params.pre_text_transform}-history-{params.num_history}"
        else:
            params.suffix += (
                f"-pre-text-{params.pre_text_transform}-history-{params.num_history}"
            )

    book_name = params.manifest_dir.split("/")[-1].replace(".jsonl.gz", "")
    setup_logger(
        f"{params.res_dir}/log-decode-{book_name}-{params.suffix}", log_level="info"
    )
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_transducer_model(params)
    tokenizer = get_tokenizer(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for" f" --iter {params.iter}, --avg {params.avg}"
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
    model.device = device

    # load manifest
    manifest = load_manifest(params.manifest_dir)

    results = []
    count = 0

    last_recording = ""
    last_end = -1
    history = []
    num_pre_texts = []

    for cut in manifest:
        if cut.has_features:
            feat = cut.load_features()
            feat_lens = cut.num_frames
        else:
            feat = cut.compute_features(extractor=Fbank())
            feat_lens = feat.shape[0]

        cur_recording = cut.recording.id

        if cur_recording != last_recording:
            last_recording = cur_recording
            history = []  # clean up the history
            last_end = -1
            logging.info("Moving on to the next recording")
        else:
            if cut.start < last_end - 0.2:  # overlap with the previous cuts
                logging.warning("An overlap exists between current cut and last cut")
                logging.warning("Skipping this cut!")
                continue
            if cut.start > last_end + 10:
                logging.warning(
                    f"Large time gap between the current and previous utterance: {cut.start - last_end}."
                )

        # prepare input
        x = torch.tensor(feat, device=device).unsqueeze(0)
        x_lens = torch.tensor(
            [
                feat_lens,
            ],
            device=device,
        )

        if params.use_pre_text:
            if params.num_history > 0:
                pre_texts = history[-params.num_history :]
            else:
                pre_texts = []
            num_pre_texts.append(len(pre_texts))
            pre_texts = [train_text_normalization(" ".join(pre_texts))]
            fixed_sentence = "Mixed-case English transcription, with punctuation. Actually, it is fully not related."
            style_texts = [fixed_sentence]

            pre_texts = _apply_style_transform(pre_texts, params.pre_text_transform)
            if params.use_style_prompt:
                style_texts = _apply_style_transform(
                    style_texts, params.style_text_transform
                )

            # encode prompts
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                encoded_inputs, style_lens = _encode_texts_as_bytes_with_tokenizer(
                    pre_texts=pre_texts,
                    style_texts=style_texts,
                    tokenizer=tokenizer,
                    device=device,
                    no_limit=True,
                )
                if params.num_history > 5:
                    logging.info(
                        f"Shape of encoded texts: {encoded_inputs['input_ids'].shape} "
                    )

                memory, memory_key_padding_mask = model.encode_text(
                    encoded_inputs=encoded_inputs,
                    style_lens=style_lens,
                )  # (T,B,C)
        else:
            memory = None
            memory_key_padding_mask = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            encoder_out, encoder_out_lens = model.encode_audio(
                feature=x,
                feature_lens=x_lens,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if params.method == "greedy_search":
            hyp_tokens = greedy_search_batch(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
        elif params.method == "modified_beam_search":
            hyp_tokens = modified_beam_search(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=params.beam_size,
            )

        hyp = sp.decode(hyp_tokens)[0]  # in string format
        ref_text = ref_text_normalization(
            cut.supervisions[0].texts[0]
        )  # required to match the training

        # extend the history
        if params.use_gt_pre_text:
            history.append(ref_text)
        else:
            history.append(hyp)
        last_end = cut.end  # update the last end timestamp

        # append the current decoding result
        hyp = hyp.split()
        ref = ref_text.split()
        results.append((cut.id, ref, hyp))

        count += 1
        if count % 100 == 0:
            logging.info(f"Cuts processed until now: {count}/{len(manifest)}")
            logging.info(
                f"Averaged context numbers of last 100 samples is: {sum(num_pre_texts[-100:])/100}"
            )

    logging.info(f"A total of {count} cuts")
    logging.info(
        f"Averaged context numbers of whole set is: {sum(num_pre_texts)/len(num_pre_texts)}"
    )

    results = sorted(results)
    recog_path = (
        params.res_dir / f"recogs-long-audio-{params.method}-{params.suffix}.txt"
    )
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")

    errs_filename = (
        params.res_dir / f"errs-long-audio-{params.method}-{params.suffix}.txt"
    )
    with open(errs_filename, "w") as f:
        wer = write_error_stats(
            f,
            f"long-audio-{params.method}",
            results,
            enable_log=True,
            compute_CER=False,
        )

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    if params.post_normalization:
        params.suffix += "-post-normalization"

        new_res = []
        for item in results:
            id, ref, hyp = item
            hyp = upper_only_alpha(" ".join(hyp)).split()
            ref = upper_only_alpha(" ".join(ref)).split()
            new_res.append((id, ref, hyp))

    new_res = sorted(new_res)
    recog_path = (
        params.res_dir
        / f"recogs-long-audio-{params.method}-{params.suffix}-post-normalization.txt"
    )
    store_transcripts(filename=recog_path, texts=new_res)
    logging.info(f"The transcripts are stored in {recog_path}")

    errs_filename = (
        params.res_dir
        / f"errs-long-audio-{params.method}-{params.suffix}-post-normalization.txt"
    )
    with open(errs_filename, "w") as f:
        wer = write_error_stats(
            f,
            f"long-audio-{params.method}",
            new_res,
            enable_log=True,
            compute_CER=False,
        )

    logging.info("Wrote detailed error stats to {}".format(errs_filename))


if __name__ == "__main__":
    main()
