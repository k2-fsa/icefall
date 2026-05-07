
import argparse
import os
import logging
from typing import Union, List, Dict
from pathlib import Path

from train_multi_KD3_shar import add_model_arguments, get_encoder_embed, get_encoder_model
from zipformer2 import Zipformer2

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse import Fbank, FbankConfig
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.input_strategies import BatchIO, OnTheFlyFeatures, PrecomputedFeatures
from lhotse.features.io import NumpyHdf5Writer
from lhotse.workarounds import Hdf5MemoryIssueFix

from icefall.utils import AttributeDict, setup_logger, make_pad_mask

class FbankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        return_cuts: bool = True,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.input_strategy = input_strategy
        
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
        
    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)
        
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl
            
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        
        batch = {
            "feature": inputs,
        }
        batch.update(supervision_intervals)
        
        if self.return_cuts:
            batch["cuts"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]
        return batch
        
class ZipformerModel(torch.nn.Module):
    def __init__(
        self, encoder_embed: torch.nn.Module, encoder: Zipformer2
    ):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        
        self.encoder_dim = encoder.encoder_dim
    
    def _get_full_dim_output_impl(self, outputs: List[torch.Tensor], max_depth):
        output_dim = max(self.encoder_dim[:max_depth])
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[max_depth - 1]
        
        for i in range(max_depth - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)
    
    def _get_full_dim_output(self, outputs: List[torch.Tensor], max_depth: int):
        outputs = outputs[:max_depth]
        return self._get_full_dim_output_impl(outputs, max_depth=max_depth)
    
    def get_embeddings(self, batch, layer_idx: int = -1):
        device = next(self.parameters()).device
        x = batch["feature"].to(device)
        x_lens = batch["num_frames"].to(device)
        
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens, layer_results = self.encoder(
            x, x_lens, src_key_padding_mask, return_middle_out=True
        )
        
        if layer_idx == -1:
            feature = encoder_out.permute(1, 0, 2)    
        else:
            # the intermediate layers' feature are 50 Hz
            feature = self._get_full_dim_output(layer_results, layer_idx)
            # feature = layer_results[layer_idx-1] # index starts from 1
            feature = feature.permute(1, 0, 2)
            encoder_out_lens = x_lens
        
        return feature, encoder_out_lens


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
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
        default="data/embeddings"
    )

    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=-1,
        help="Which layer's representation should be extracted, index start from 1, i.e the 10-th layer requires"
        "--embedding-layer 10"
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
        help="Where to store the manifest augmented with whisper features"
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

@torch.no_grad()
def extract_embeddings(
    rank: int,
    manifest: str,
    params: AttributeDict,
):
    setup_logger(f"data/embeddings/log/log-zipformer-embeddings")
    if params.num_jobs > 1:
        manifest = manifest[rank]
        output_manifest = params.embedding_dir / f"zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}.jsonl.gz"
        embedding_path = params.embedding_dir / f'zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}-{rank}'
    else:
        output_manifest = params.embedding_dir / f"zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}.jsonl.gz"
        embedding_path =  params.embedding_dir / f'zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}'
    
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
    
    with NumpyHdf5Writer(embedding_path) as writer:
        logging.info(f"Writing zipformer embeddings to {embedding_path}")
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            
            with torch.cuda.amp.autocast(enabled=True):
                embeddings, embedding_lens = model.get_embeddings(
                    batch=batch,
                    layer_idx=params.embedding_layer # which layer's embedding to be stored
                )
            embeddings = embeddings.detach().to("cpu").numpy()
            
            for idx, cut in enumerate(cuts):    
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.embedding = writer.store_array(
                    key=cut.id,
                    value=embeddings[idx][: embedding_lens[idx]],
                    temporal_dim=0,
                    frame_shift=params.frame_shift,
                    start=cut.start,
                )
                new_cuts.append(new_cut)
                num_cuts += 1
            if num_cuts and i % 100 == 0:
                logging.info(f"Cuts processed until now: {num_cuts}")
                
    logging.info(f"Finished extracting zipformer embeddings, processed a total of {num_cuts} cuts.")
                
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
        ori_cut.embedding = embed_cut.embedding
    
    input_cuts.to_jsonl(output_dir)
    print(f"Saved the joined manifest to {output_dir}")
    
def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 29.9:
        return False
    return True

def remove_sp(c):
    if "sp1.1" in c.id or "sp0.9" in c.id:
        return False
    return True


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict()
    params.update(vars(args))
    params.embedding_dir = Path(params.embedding_dir)
    
    nj = params.num_jobs
    cuts = load_manifest(params.input_manifest)
    cuts = cuts.filter(remove_short_and_long_utt) # remove audio longer than 30s
    cuts = cuts.filter(remove_sp) # remove the speed perturbed audio
    print(f"Finished loading manifest")
    
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
            print(f"Finished splitting manifest")
            mp.spawn(extract_embeddings, args=(splitted_cuts, params), nprocs=nj, join=True)
            manifests =  f"{str(params.embedding_dir)}/zipformer-{params.zipformer_version}-layer-{params.embedding_layer}-{params.manifest_name}-*.jsonl.gz"
            os.system(f"lhotse combine {manifests} {embedding_manifest}")
    else:
        print(f"Skip embedding extraction: the manifest is already generated.")
    
    output_manifest = params.target_manifest_file
    if not os.path.exists(output_manifest):
        join_manifests(
            input_cuts=cuts,
            embedding_manifest=embedding_manifest,
            output_dir=output_manifest,
        )
    
    