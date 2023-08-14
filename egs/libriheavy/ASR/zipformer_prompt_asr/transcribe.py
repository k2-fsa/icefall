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
from lhotse import load_manifest

from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from text_normalization import (
    ref_text_normalization,
    remove_non_alphabetic,
    upper_only_alpha,
    upper_all_char,
    lower_all_char,
    lower_only_alpha,
    train_text_normalization,
)
from train_bert_encoder import (
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
        "--manifest-dir",
        type=str,
        default="data/long_audios/long_audio_pomonastravels_combined.jsonl.gz",
        help="""This is the manfiest for long audio transcription. 
        It is intended to be sored, i.e first sort by recording ID and then sort by
        start timestamp"""
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
        "--pre-text-transform",
        type=str,
        choices=["mixed-punc", "upper-no-punc", "lower-no-punc","lower-punc"],
        default="mixed-punc",
        help="The style of content prompt, i.e pre_text"
    )
    
    parser.add_argument(
        "--num-history",
        type=int,
        default=2,
        help="How many previous chunks to look if using pre-text for decoding"
    )
    
    parser.add_argument(
        "--use-gt-pre-text",
        type=str2bool,
        default=False,
        help="Whether use gt pre text when using content prompt",
    )
    
    add_model_arguments(parser)
    
    return parser
    
def _apply_style_transform(text: List[str], transform: str) -> List[str]:
    """Apply transform to a list of text. By default, the text are in 
    ground truth format, i.e mixed-punc.

    Args:
        text (List[str]): Input text string
        transform (str): Transform to be applied

    Returns:
        List[str]: _description_
    """
    if transform == "mixed-punc":
        return text
    elif transform == "upper-no-punc":
        return [upper_only_alpha(s) for s in text]
    elif transform == "lower-no-punc":
        return [lower_only_alpha(s) for s in text]
    elif transform == "lower-punc":
        return [lower_all_char(s) for s in text]
    else:
        raise NotImplementedError(f"Unseen transform: {transform}")
    

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
        params.suffix += (
            f"-{params.method}-beam-size-{params.beam_size}"
        )
        
    if params.use_pre_text:
        if params.use_gt_pre_text:
            params.suffix += f"-use-gt-pre-text-{params.pre_text_transform}-history-{params.num_history}"
        else:
            params.suffix += f"-pre-text-{params.pre_text_transform}-history-{params.num_history}"
        
        
    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}", log_level="info")
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
    
    # load manifest
    manifest = load_manifest(params.manifest_dir)

    results = []
    count = 0
    
    last_recording = ""
    last_end = -1
    history = []
    for cut in manifest:
        feat = cut.load_features()
        feat_lens = cut.num_frames
        
        cur_recording = cut.recording.id
        
        if cur_recording != last_recording:
            last_recording = cur_recording
            history = [] # clean history
            last_end = -1
        else:
            if cut.start < last_end: # overlap exits
                logging.warning(f"An overlap exists between current cut and last cut")
        
        # prepare input
        x = torch.tensor(feat, device=device).unsqueeze(0)
        x_lens = torch.tensor([feat_lens,], device=device)
        
        if params.use_pre_text:
            pre_texts = history[-params.num_history:]
            pre_texts = [train_text_normalization(" ".join(pre_texts))]
            
            if len(pre_texts) > 1000:
                pre_texts = pre_texts[-1000:]
            
            pre_texts = _apply_style_transform(pre_texts, params.pre_text_transform)
            
            # encode pre_text
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
            
        hyp = sp.decode(hyp_tokens)[0] # in string format
        ref_text = ref_text_normalization(cut.supervisions[0].texts[0]) # required to match the training
        
        # extend the history, the history here is in original format
        if params.use_gt_pre_text:
            history.append(ref_text) 
        else:
            history.append(hyp)
        last_end = cut.end # update the last end timestamp
        
        # append the current decoding result
        ref = remove_non_alphabetic(ref_text.upper(), strict=True).split() # split
        ref = [w for w in ref if w != ""]
        hyp = remove_non_alphabetic(hyp.upper(), strict=True).split() # split 
        hyp = [w for w in hyp if w != ""]
        results.append((cut.id, ref, hyp))

        count += 1
        if count % 100 == 0:
            logging.info(f"Cuts processed until now: {count}/{len(manifest)}")

    results = sorted(results)
    recog_path = params.res_dir / f"recogs-long-audio-{params.method}-{params.suffix}.txt"
    store_transcripts(filename=recog_path, texts=results)
    logging.info(f"The transcripts are stored in {recog_path}")
    
    errs_filename = params.res_dir / f"errs-long-audio-{params.method}-{params.suffix}.txt"
    with open(errs_filename, "w") as f:
        wer = write_error_stats(
            f, f"long-audio-{params.method}", results, enable_log=True, compute_CER=False,
        )

    logging.info("Wrote detailed error stats to {}".format(errs_filename))
    
if __name__=="__main__":
    main()