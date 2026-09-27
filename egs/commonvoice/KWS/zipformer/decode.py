#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Decode Common Voice Japanese as phone tokens and report PER.

The maintained WenetSpeech KWS decoder supplies the streaming encoder and
RNN-T search implementation.  This adapter replaces its lexicon and corpus
assumptions with the phone-only Common Voice recipe contract.
"""

import importlib.util
import logging
from collections import defaultdict
from pathlib import Path
import sys

ICEFALL_ROOT = Path(__file__).resolve().parents[4]
if str(ICEFALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ICEFALL_ROOT))

import k2
import torch
from lhotse.cut import Cut

from icefall.checkpoint import (  # noqa: E402
    average_checkpoints,
    average_checkpoints_with_averaged_model,
)

from egs.commonvoice.KWS.zipformer.asr_datamodule import (  # noqa: E402
    CommonVoiceKwsDataModule,
    has_plausible_transcript_rate,
)

KWS_LOCAL = Path(__file__).resolve().parents[1] / "local"
if str(KWS_LOCAL) not in sys.path:
    sys.path.insert(0, str(KWS_LOCAL))

from japanese_phones import text_to_phones  # noqa: E402

WENETSPEECH_ZIPFORMER = (
    Path(__file__).resolve().parents[3] / "wenetspeech" / "KWS" / "zipformer"
)
if str(WENETSPEECH_ZIPFORMER) not in sys.path:
    sys.path.insert(0, str(WENETSPEECH_ZIPFORMER))

spec = importlib.util.spec_from_file_location(
    "wenetspeech_kws_decode_asr", WENETSPEECH_ZIPFORMER / "decode-asr.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {WENETSPEECH_ZIPFORMER / 'decode-asr.py'}")
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class PhoneLexicon:
    """The subset of ``icefall.lexicon.Lexicon`` needed for phone decoding."""

    def __init__(self, lang_dir: Path):
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.tokens = sorted(
            self.token_table[symbol]
            for symbol in self.token_table.symbols
            if self.token_table[symbol] != 0
        )


def get_parser():
    parser = decoder.get_parser()
    CommonVoiceKwsDataModule.add_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "Optional single raw checkpoint beneath --exp-dir. When empty, "
            "--epoch/--avg/--use-averaged-model select the model."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("dev", "test"),
        default="test",
        help="Common Voice split to decode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PER output directory; defaults to <exp-dir>/per.",
    )
    return parser


def load_model_for_decoding(params, model, device):
    """Load a raw checkpoint or Icefall's standard epoch-averaged model."""

    if params.checkpoint:
        checkpoint = params.exp_dir / params.checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        logging.info("Loading the raw checkpoint %s", checkpoint)
        decoder.load_checkpoint(checkpoint, model)
        return

    if params.avg <= 0:
        raise ValueError(f"--avg must be positive, got {params.avg}")

    if params.use_averaged_model:
        start = params.epoch - params.avg
        if start < 1:
            raise ValueError(
                f"epoch {params.epoch} with avg {params.avg} needs epoch-{start}.pt"
            )
        filename_start = params.exp_dir / f"epoch-{start}.pt"
        filename_end = params.exp_dir / f"epoch-{params.epoch}.pt"
        for filename in (filename_start, filename_end):
            if not filename.is_file():
                raise FileNotFoundError(filename)
        logging.info(
            "Calculating the averaged model over epochs %s (excluded) to %s",
            start,
            params.epoch,
        )
        state_dict = average_checkpoints_with_averaged_model(
            filename_start=filename_start,
            filename_end=filename_end,
            device=device,
        )
    elif params.avg == 1:
        checkpoint = params.exp_dir / f"epoch-{params.epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        decoder.load_checkpoint(checkpoint, model)
        return
    else:
        start = params.epoch - params.avg + 1
        filenames = [
            params.exp_dir / f"epoch-{epoch}.pt"
            for epoch in range(start, params.epoch + 1)
        ]
        for filename in filenames:
            if not filename.is_file():
                raise FileNotFoundError(filename)
        logging.info("Averaging raw checkpoints %s", filenames)
        state_dict = average_checkpoints(filenames, device=device)

    model.load_state_dict(state_dict)


def remove_short_utterance(cut: Cut) -> bool:
    num_encoder_frames = ((cut.num_frames - 7) // 2 + 1) // 2
    if num_encoder_frames <= 0:
        logging.warning(
            "Exclude cut %s from decoding: only %s input frames",
            cut.id,
            cut.num_frames,
        )
    return num_encoder_frames > 0


def decode_dataset(dl, params, model, lexicon):
    """Decode once and retain both official and transcript-filtered results."""

    all_results = defaultdict(list)
    filtered_results = defaultdict(list)
    num_cuts = 0
    num_filtered = 0

    for batch_idx, batch in enumerate(dl):
        cuts = batch["supervisions"]["cut"]
        references = [text_to_phones(cut.supervisions[0].text) for cut in cuts]
        hypotheses = decoder.decode_one_batch(
            params=params,
            model=model,
            lexicon=lexicon,
            graph_compiler=None,
            decoding_graph=None,
            batch=batch,
        )

        for setting, hyps in hypotheses.items():
            if len(hyps) != len(references):
                raise RuntimeError(
                    f"Decoder returned {len(hyps)} hypotheses for "
                    f"{len(references)} references"
                )
            for cut, ref_phones, hyp_phones in zip(cuts, references, hyps):
                item = (cut.id, ref_phones, hyp_phones)
                all_results[setting].append(item)
                if has_plausible_transcript_rate(cut):
                    filtered_results[setting].append(item)

        num_cuts += len(cuts)
        num_filtered += sum(has_plausible_transcript_rate(cut) for cut in cuts)
        if batch_idx % 20 == 0:
            logging.info("batch %s, decoded %s cuts", batch_idx, num_cuts)

    logging.info(
        "Decoded %s official %s cuts; %s remain after transcript-rate filter",
        num_cuts,
        params.split,
        num_filtered,
    )
    return all_results, filtered_results


def save_per_results(params, test_set_name, results_dict):
    per_by_setting = {}
    for setting, results in results_dict.items():
        results = sorted(results)
        recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        decoder.store_transcripts(filename=recog_path, texts=results)

        error_path = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with error_path.open("w", encoding="utf-8") as stream:
            per_by_setting[setting] = decoder.write_error_stats(
                stream,
                f"{test_set_name}-{setting}",
                results,
                enable_log=True,
            )

    summary_path = params.res_dir / f"per-summary-{test_set_name}-{params.suffix}.txt"
    with summary_path.open("w", encoding="utf-8") as stream:
        print("settings\tPER", file=stream)
        for setting, per in sorted(per_by_setting.items(), key=lambda item: item[1]):
            print(f"{setting}\t{per}", file=stream)
    logging.info("Wrote PER summary to %s", summary_path)


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = decoder.get_params()
    params.update(vars(args))
    if params.decoding_method not in ("greedy_search", "modified_beam_search"):
        raise ValueError(
            "The phone PER adapter supports greedy_search and "
            "modified_beam_search"
        )

    params.res_dir = params.output_dir or params.exp_dir / "per"
    params.res_dir.mkdir(parents=True, exist_ok=True)
    if params.checkpoint:
        model_suffix = Path(params.checkpoint).stem
    else:
        model_suffix = f"epoch-{params.epoch}-avg-{params.avg}"
        if params.use_averaged_model:
            model_suffix += "-use-averaged-model"
    params.suffix = f"{model_suffix}-{params.decoding_method}"
    if params.decoding_method == "modified_beam_search":
        params.suffix += f"-beam-{params.beam_size}"
    params.suffix += f"-blank-penalty-{params.blank_penalty}"
    decoder.setup_logger(params.res_dir / f"log-decode-{params.suffix}")

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    lexicon = PhoneLexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    logging.info("Device: %s", device)
    logging.info(params)
    model = decoder.get_model(params)
    model.to(device)
    load_model_for_decoding(params, model, device)
    model.eval()
    logging.info("Number of model parameters: %s", sum(p.numel() for p in model.parameters()))

    args.return_cuts = True
    data_module = CommonVoiceKwsDataModule(args)
    if params.split == "dev":
        cuts = data_module.dev_cuts()
    else:
        cuts = data_module.test_cuts()
    cuts = cuts.filter(remove_short_utterance)
    dl = data_module.test_dataloaders(cuts)

    official, filtered = decode_dataset(dl, params, model, lexicon)
    split_name = params.split.upper()
    save_per_results(params, split_name, official)
    save_per_results(params, f"{split_name}_FILTERED", filtered)
    logging.info("Done!")


if __name__ == "__main__":
    main()
