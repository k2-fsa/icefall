#!/usr/bin/env python3

import argparse
import logging

import k2
import kaldifeat
import torch
import torchaudio
from conformer import Conformer

from icefall.decode import (
    get_lattice,
    one_best_decoding,
)
from icefall.utils import AttributeDict, get_texts


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint."
        "The checkpoint is assume to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--words-file",
        type=str,
        required=True,
        help="Path to words.txt",
    )

    parser.add_argument(
        "--hlg", type=str, required=True, help="Path to HLG.pt."
    )

    parser.add_argument(
        "--sound-file",
        type=str,
        required=True,
        help="The input sound file to transcribe. "
        "Supported formats are those that supported by torchaudio.load(). "
        "For example, wav, flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "feature_dim": 80,
            "nhead": 8,
            "num_classes": 5000,
            "attention_dim": 512,
            "subsampling_factor": 4,
            "num_decoder_layers": 6,
            "vgg_frontend": False,
            "is_espnet_structure": True,
            "mmi_loss": False,
            "use_feat_batchnorm": True,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=params.num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=params.vgg_frontend,
        is_espnet_structure=params.is_espnet_structure,
        mmi_loss=params.mmi_loss,
        use_feat_batchnorm=params.use_feat_batchnorm,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    HLG = k2.Fsa.from_dict(torch.load(params.hlg))
    HLG = HLG.to(device)

    model.to(device)

    wave, samp_freq = torchaudio.load(params.sound_file)
    wave = wave.squeeze().to(device)

    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = samp_freq
    opts.mel_opts.num_bins = 80

    fbank = kaldifeat.Fbank(opts)

    features = fbank(wave)
    features = features.unsqueeze(0)

    nnet_output, _, _ = model(features)
    supervision_segments = torch.tensor(
        [[0, 0, nnet_output.shape[1]]], dtype=torch.int32
    )

    lattice = get_lattice(
        nnet_output=nnet_output,
        HLG=HLG,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    best_path = one_best_decoding(
        lattice=lattice, use_double_scores=params.use_double_scores
    )

    hyps = get_texts(best_path)
    word_sym_table = k2.SymbolTable.from_file(params.words_file)
    hyps = [[word_sym_table[i] for i in ids] for ids in hyps]
    logging.info(hyps)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
