import argparse
import logging
import math
import warnings
from pathlib import Path
from typing import List
from tqdm import tqdm

import k2
import kaldifeat
import sentencepiece as spm
import torch
import torchaudio
from lhotse import load_manifest, Fbank

from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from text_normalization import ref_text_normalization, remove_non_alphabetic, upper_only_alpha, upper_all_char, lower_all_char, lower_only_alpha
from train_bert_encoder import (
    add_model_arguments,
    get_params,
    get_tokenizer,
    get_transducer_model,
    _encode_texts_as_bytes,
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
        "--manifest-dir",
        type=str,
        default="data/manifests_chunk/youtube_cuts_foxnews.jsonl.gz"
    )
    
    parser.add_argument(
        "--segment-length",
        type=float,
        default=30.0,
    )
    
    parser.add_argument(
        "--use-pre-text",
        type=str2bool,
        default=False,
        help="Whether use pre-text when decoding the current chunk"
    )
    
    parser.add_argument(
        "--num-history",
        type=int,
        default=2,
        help="How many previous chunks to look if using pre-text for decoding"
    )
    
    add_model_arguments(parser)
    
    return parser
    
def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. "
            f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans

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

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
        
    if "beam_search" in params.decoding_method:
        params.suffix += (
            f"-{params.decoding_method}-beam-size-{params.beam_size}"
        )

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"
        
    if params.use_pre_text:
        params.suffix += f"-pre-text-{params.pre_text_transform}"
        
    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
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
    model.device = device
    
    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = params.feature_dim

    params.res_dir = params.exp_dir / "long_audio_transcribe"
    params.res_dir.mkdir(exist_ok=True)
    
    
    # load manifest
    manifest = load_manifest(params.manifest_dir)
    fbank = kaldifeat.Fbank(opts)

    all_hyps = []
    all_ref = []
    results = []
    count = 0
    
    for cut in tqdm(manifest):
        frames_per_segment = params.segment_length * 100 # number of frames per segment
        
        feats = cut.compute_features(extractor=Fbank())
        feats = torch.tensor(feats).to(device)
        
        num_chunks = feats.size(0) // frames_per_segment + 1
        hyp = []
        for i in range(int(num_chunks)):
            start = int(i * frames_per_segment)
            end = int(min((i+1) * frames_per_segment, feats.size(0)))
            x = feats[start:end].unsqueeze(0)
            x_lens = torch.tensor([end-start,], device=device)
            
            if params.use_pre_text:
                pre_texts = hyp[-params.num_history:]
                pre_texts = [" ".join(pre_texts)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    encoded_inputs = tokenizer(
                        pre_texts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=500,
                    ).to(device)
                    
                    memory, memory_key_padding_mask = model.encode_text(
                        encoded_inputs=encoded_inputs,
                    ) # (T,B,C)
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

            hyp_tokens = greedy_search_batch(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
            
            hyp.append(sp.decode(hyp_tokens)[0]) 
        
        ref = remove_non_alphabetic(cut.text.upper())
        hyp = remove_non_alphabetic(" ".join(hyp).upper())
        all_hyps.append(hyp)
        all_ref.append(ref)
        results.append((cut.id, ref, hyp))
        count += 1
        if count == 5:
            break
    
    recog_path = params.res_dir / f"recogs-youtube-{params.method}-{params.suffix}.txt"
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")
    
    errs_filename = params.res_dir / f"errs-youtube-{params.method}-{params.suffix}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(
            f, f"youtube-{params.method}", results, enable_log=True
        )

    logging.info("Wrote detailed error stats to {}".format(errs_filename))


    
    
if __name__=="__main__":
    main()