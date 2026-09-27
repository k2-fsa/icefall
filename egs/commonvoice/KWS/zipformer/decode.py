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
        default="best-valid-loss.pt",
        help="Checkpoint filename beneath --exp-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PER output directory; defaults to <exp-dir>/per.",
    )
    return parser


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
        "Decoded %s official test cuts; %s remain after transcript-rate filter",
        num_cuts,
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
    params.suffix = f"{Path(params.checkpoint).stem}-{params.decoding_method}"
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
    checkpoint = params.exp_dir / params.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    decoder.load_checkpoint(checkpoint, model)
    model.to(device)
    model.eval()
    logging.info("Number of model parameters: %s", sum(p.numel() for p in model.parameters()))

    args.return_cuts = True
    data_module = CommonVoiceKwsDataModule(args)
    test_cuts = data_module.test_cuts().filter(remove_short_utterance)
    test_dl = data_module.test_dataloaders(test_cuts)

    official, filtered = decode_dataset(test_dl, params, model, lexicon)
    save_per_results(params, "TEST", official)
    save_per_results(params, "TEST_FILTERED", filtered)
    logging.info("Done!")


if __name__ == "__main__":
    main()
