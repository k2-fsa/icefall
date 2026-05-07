
import argparse
import os
import io
import logging
from pathlib import Path

from icefall.utils import AttributeDict, setup_logger, str2bool

from train_multi_KD3_shar import add_model_arguments, get_encoder_embed, get_encoder_model
from collect_zipformer_embeddings import FbankDataset, ZipformerModel

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import lhotse
from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse import Fbank, FbankConfig
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fastcopy
import multi_quantization as quantization
import numpy as np

from typing import Union, Optional

lhotse.set_caching_enabled(True)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # quantizer related
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
    )
    
    parser.add_argument(
        "--num-cb",
        type=int,
        default=4,
    )
    
    parser.add_argument(
        "--quantizer-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--s3-prefix",
        type=str,
        required=True,
        default="brainllm:s3://yangxiaoyu/LibriSpeech"
    )
    
    # others
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--manifest-name",
        type=str,
        required=True,
        help="name of the manifest, e.g embeddings-dev-clean, embeddings-train-clean-100"
    )
    
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/vq_whisper"
    )

    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=-1,
        help="Which layer's representation should be extracted",
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--target-manifest-file",
        type=str,
        required=True,
        help="Where to store the manifest augmented with zipformer features"
    )
    
    parser.add_argument(
        "--normalize",
        type=str2bool,
        default=False,
        help="If True, compute the channel-wise mean and std on the training se for nomalization."
    )
    
    # zipformer related args
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--zipformer-version",
        type=str,
        default="300m",
    )
    
    parser.add_argument(
        "--frame-shift",
        type=float,
        default=0.02,
    )
    
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=128,
    )
    add_model_arguments(parser)
    
    return parser

def normalize_data(data, mean, std):
    return (data - mean) / (std + 1e-5)

@torch.no_grad()
def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/vq_zipformer_client/log/log-zipformer-cb-indexes")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}.jsonl.gz"
    else:
        output_manifest = params.embedding_dir / f"zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
    
    device = torch.device("cuda", rank)
    
    # currently only use the encoder of zipformer
    logging.info(params)
    model = ZipformerModel(
        encoder_embed=get_encoder_embed(params),
        encoder=get_encoder_model(params),
    )
    state_dict = torch.load(params.model_ckpt)["model"]
    load_info = model.load_state_dict(state_dict, strict=False)
    logging.info(load_info)
    
    model.to(device)
    model.eval()
    logging.info(f"Number of zipformer model params: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"Successfully loaded zipformer model.")
    
    quantizer = quantization.Quantizer(
        dim=params.embedding_dim,
        num_codebooks=params.num_cb,
        codebook_size=256,
    )
    state_dict = torch.load(params.quantizer_path)
    if "quantizer" not in state_dict:
        # with out normalization stats
        assert not params.normalize, "No normalization stats is found!"
        state_dict = {"quantizer": state_dict}
    
    if params.normalize:
        mu = state_dict["mean"].to(device)
        std = state_dict["std"].to(device)
    quantizer.load_state_dict(state_dict["quantizer"])
    quantizer.to(device)
    
    dataset = FbankDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=128))),
        return_cuts=True
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=params.max_duration,
        shuffle=False,
        num_buckets=20,
        buffer_size=20 * 2000,
        shuffle_buffer_size=20 * 5000,
        drop_last=False,
    )
    
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=2,
        persistent_workers=False,
    )
    
    new_cuts = []
    num_cuts = 0
    
    logging.info(f"Writing zipformer indexes")
    for i, batch in enumerate(dl):
        cuts = batch["cuts"]
        
        with torch.cuda.amp.autocast(enabled=True):
            embeddings, embedding_lens = model.get_embeddings(
                batch,
                layer_idx=params.embedding_layer # which layer's embedding to be stored
            )
        embeddings = embeddings.float()
        if params.normalize:
            embeddings = normalize_data(embeddings, mu, std)
        
        # codebook_indexes = quantizer.encode(embeddings) # [N, T, C]
        N,T,C = embeddings.shape
        embeddings = embeddings.reshape(-1, C)
        B = 2000
        splits = embeddings.split(B)
        codebook_indexes = []
        for chunk in splits:
            chunk_indexes = quantizer.encode(chunk)
            codebook_indexes.append(chunk_indexes)
        codebook_indexes = torch.cat(codebook_indexes).reshape(N,T,params.num_cb)
        codebook_indexes = codebook_indexes.to("cpu").numpy()
        assert np.min(codebook_indexes) >= 0
        assert np.max(codebook_indexes) < 256
        
        for idx, cut in enumerate(cuts):
            cb_index = codebook_indexes[idx][: embedding_lens[idx]]
            
            if "/" in cut.id:
                # we are dealing with libriheavy cuts
                filename = cut.id
            else:
                filename = "/".join(cut.id.split("-")[:2]) + "/" + cut.id
            output_path = f"{params.s3_prefix}/{filename}.npy"
            if os.path.exists(output_path):
                logging.info(f"This codebook file has already been generated. Please check if you are doing correctly!")
                
            base_dir, filename = output_path.rsplit("/", 1)
            os.makedirs(base_dir, exist_ok=True)
            np.save(output_path, cb_index)
                
            info = {
                "path": output_path,
                "shape": list(cb_index.shape),
                "frame-shift": params.frame_shift,
            }
            
            new_cut = fastcopy(
                cut,
                custom={"codebook_indexes": info}
            )
            new_cuts.append(new_cut)
            num_cuts += 1
            if num_cuts and num_cuts % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
                
    logging.info(f"Finished extracting zipformer codebook indexes, processed a total of {num_cuts} cuts.")
                
    CutSet.from_cuts(new_cuts).to_jsonl(output_manifest)
    logging.info(f"Saved manifest to {output_manifest}")
    
def join_manifests(
    input_cuts: CutSet,
    embedding_manifest: str,
    output_dir: str,
):
    # Combine the teacher embedding manifest with the original manifest for ASR
    embedding_cuts = load_manifest(embedding_manifest)
    
    assert len(embedding_cuts) == len(input_cuts)
    assert set(input_cuts.ids) == set(embedding_cuts.ids)
    
    embedding_cuts = embedding_cuts.sort_like(input_cuts)
    for cut_idx, (ori_cut, embed_cut) in enumerate(zip(input_cuts, embedding_cuts)):
        assert ori_cut.id == embed_cut.id
        ori_cut.codebook_indexes = embed_cut.codebook_indexes
    
    input_cuts.to_jsonl(output_dir)
    logging.info(f"Saved the joined manifest to {output_dir}")
    
def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 29.9:
        return False
    return True

def remove_sp(c):
    if "sp0.9" in c.id or "sp1.1" in c.id:
        return False
    return True

if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.embedding_dir = Path(params.embedding_dir)
    
    nj = params.num_jobs
    print(f"Start loading manifest")
    cuts = load_manifest(params.input_manifest)
    cuts = cuts.filter(remove_short_and_long_utt) # remove audio longer than 30s
    cuts = cuts.filter(remove_sp) # remove speed perturb
    print(f"Finished loading manifest")
    print(cuts)
    
    embedding_manifest = params.embedding_dir / f"zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
    
    if not embedding_manifest.exists():
        if nj == 1:
            extract_embeddings(
                rank=0,
                manifest=cuts,
                params=params,    
            )
        else:
            splitted_cuts = cuts.split(num_splits=nj)
            logging.info(f"Finished splitting manifest")
            mp.spawn(extract_embeddings, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {embedding_manifest}")
    else:
        logging.info(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.target_manifest_file
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=embedding_manifest,
            output_dir=output_manifest,
        )
    
    