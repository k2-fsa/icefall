#!/usr/bin/env python3

import argparse
import logging
import k2
import torch
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)
from lhotse.cut import Cut
from train_s import add_model_arguments, get_model, get_params
from icefall.utils import add_sos, make_pad_mask, time_warp, AttributeDict, str2bool
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from scaling import (
    ActivationDropoutAndLinear,
    Balancer,
    BiasNorm,
    ChunkCausalDepthwiseConv1d,
    Dropout2,
    FloatLike,
    ScheduledFloat,
    Whiten,
    convert_num_channels,
    limit_param_value,
    penalize_abs_values_gt,
    softmax,
)
from typing import Dict, List, Tuple
from pathlib import Path
from icefall.lexicon import Lexicon

def _load_zipformer_model(params: AttributeDict) -> nn.Module:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    lexicon = Lexicon(params.lang_dir)
    params.blank_id = lexicon.token_table["<blk>"]
    params.vocab_size = max(lexicon.tokens) + 1

    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )

    # logging.info(params)

    logging.info("About to create model")
    model = get_model(params)


    load_checkpoint(f"{params.kd_exp_dir}/{params.teacher_model_id}.pt", model)
    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    return model


class ZipformerS:
    """
    A wrapper of zipformer-l model.

    A teacher model is responsible for:
        1. load teacher model
        2. extracting embeddings to train quantizer.
        3. extract codebook indices
        4. verify its performance with ctc_greedy_search method.
    """

    def __init__(self, params: AttributeDict):
        self.model =  _load_zipformer_model(params)
        self.encoder_embed = self.model.encoder_embed
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.joiner = self.model.joiner
        self.params = params


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        # Options about model loading.
        parser.add_argument(
            "--zipformer-model-dir",
            type=Path,
            default="./mvq_kd_zipformer/exp/zipformer_models/",
            help="path to save downloaded zipformer models.",
        )

        parser.add_argument(
            "--teacher-model-id",
            type=str,
            default="zipformer_l_56",
            help="could be one of:  zipformer_l_56",
        )
        parser.add_argument(
            "--total-layers",
            type=int,
            default=7,
        )
        parser.add_argument(
            "--kd-exp-dir",
            type=Path,
            default="mvq_kd_zipformer/exp/",
            help="The experiment dir",
        )

        parser.add_argument(
            "--use-extracted-codebook",
            type=str2bool,
            default=False,
            help="Whether to use the extracted codebook indexes.",
        )

        parser.add_argument(
            "--ref-duration",
            type=float,
            default=600,
            help="""Reference batch duration for purposes of adjusting batch counts for setting various schedules inside the model""",
        )

        parser.add_argument(
            "--embedding-dim",
            type=int,
            default=192,
            help="""parameters used by quantizer""",
        )
        print("ZipformerL executed add_arguments ... ")

    @staticmethod
    def get_params() -> AttributeDict:
        """Return a dict containing parameters defined in other modules.

        Their default value conflits to hubert's requirements so they are reset as following.
        """
        params = AttributeDict(
            {
                # parameters defined in asr_datamodule.py
                # "input_strategy": "AudioSamples",
                # "enable_musan": False,
                # "enable_spec_aug": False,
                "return_cuts": True,
                "drop_last": False,
                # parameters used by quantizer
                # "embedding_dim": 1280,
                # "embedding_dim": 192,
            }
        )
        return params


    # Modified from Zipformer2.forward to extract all middle layers output
    def extract_layers_result(
        self,
        batch: Dict,
    ) -> List[torch.Tensor]:
        """
        Extract activations from all layers.
        """
        features = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert features.ndim == 3
        features = features.to(self.params.device)
        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(self.params.device)

        #encoder_embed  0~6
        assert isinstance(self.params.embedding_layer, int), f"embedding_layer type :  {type(self.params.embedding_layer).__name__}"

        if self.params.embedding_layer == 0:  #extract encoder_embed
            print("input feature shape : ", features.shape)
            features, _ = self.encoder_embed.forward(x = features, x_lens = feature_lens)
            print(f"extracted encoder_embed features...    embedding_layer : {self.params.embedding_layer}", features.shape)

        elif self.params.embedding_layer > 0  and  self.params.embedding_layer < 7:  #extract Zipformer2Encoder and DownsampledZipformer2Encoder
            features, feature_lens = self.encoder_embed.forward(x = features, x_lens = feature_lens)
            src_key_padding_mask = make_pad_mask(feature_lens)
            features = features.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
            #expand this """   encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)   """

            outputs = []
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                feature_masks = [1.0] * len(self.encoder.encoder_dim)
            else:
                feature_masks = self.encoder.get_feature_masks(features)

            chunk_size, left_context_chunks = self.encoder.get_chunk_info()

            if torch.jit.is_scripting() or torch.jit.is_tracing():
                # Not support exporting a model for simulating streaming decoding
                attn_mask = None
            else:
                attn_mask = self.encoder._get_attn_mask(features, chunk_size, left_context_chunks)

            for i, module in enumerate(self.encoder.encoders[:self.params.embedding_layer]):
                ds = self.encoder.downsampling_factor[i]
                x = convert_num_channels(features, self.encoder.encoder_dim[i])

                x = module(
                    x,
                    chunk_size=chunk_size,
                    feature_mask=feature_masks[i],
                    src_key_padding_mask=(
                        None
                        if src_key_padding_mask is None
                        else src_key_padding_mask[..., ::ds]
                    ),
                    attn_mask=attn_mask,
                )
                outputs.append(x)

            features = outputs[-1]

            #expand this """   encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)   """
            print(f"extracted Zipformer2Encoder and DownsampledZipformer2Encoder features...   embedding_layer : {self.params.embedding_layer}", features.shape)
        
        elif self.params.embedding_layer == 7:  #extract encoder
            features, encoder_out_lens, middle_layer_outputs = self.model.forward_encoder(x = features, x_lens = feature_lens)

        else:
            print("param embedding_layer invalid. ", features.shape)
            return

        return features


    def extract_embedding(self, batch) -> Tuple[torch.tensor, List[int]]:
        """
        Eextract embeddings specified by self.params.embedding_layer.

        These embeddings could be used to train quantizer
        or to extract codebook indexes.

        The returned List[int] is valid length of each embedding.
        We only want to store codebook indexes related to
        these valid embeddings.
        """
        supervisions = batch["supervisions"]
        cut_list = supervisions["cut"]
        assert all(c.start == 0 for c in cut_list)
        if self.params.embedding_layer == 0: 
            embeddings = self.extract_layers_result(batch)
            N = embeddings.shape[0]
            assert len(cut_list) == N
            num_frames = (supervisions["num_frames"] // 2).tolist()
            return embeddings, num_frames

        elif self.params.embedding_layer > 0  and  self.params.embedding_layer < 7:
            embeddings = self.extract_layers_result(batch)
            encoder_embedding = embeddings.permute(1, 0, 2)  # N, T, C   batch_size,  time_frame,  fbank_dimension
            N = encoder_embedding.shape[0]
            assert len(cut_list) == N
            # 320 is from: 16,000 / 50 = sample_rate / hbuert output frame rate
            # inputs are 100 frame rate, middle layer outputs are 50 frame rate, 100 / 50 = 2
            num_frames = (supervisions["num_frames"] // 2).tolist()
            return encoder_embedding, num_frames

        elif self.params.embedding_layer == 7: 
            embeddings = self.extract_layers_result(batch)
            N = embeddings.shape[0]
            assert len(cut_list) == N
            num_frames = (supervisions["num_frames"] // 4).tolist()
            return embeddings, num_frames

        else:
            print (" param embedding_layer invalid. ", self.params.embedding_layer)
            return
