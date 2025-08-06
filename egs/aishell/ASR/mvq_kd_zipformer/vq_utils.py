#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Author: Liyong Guo)
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


import argparse
import copy
import glob
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import List, Tuple

model_type = os.environ.get("MODEL_TYPE")
# 检查环境变量是否存在
if model_type is None:
    print("错误：MODEL_TYPE 环境变量未设置！", file=sys.stderr)
    sys.exit(1)  # 非零退出码表示异常退出

import numpy as np
import torch
import torch.multiprocessing as mp

from icefall import is_module_available

if not is_module_available("multi_quantization"):
    raise ValueError("Please 'pip install multi_quantization' first.")

import multi_quantization as quantization
from asr_datamodule import AishellAsrDataModule
if model_type == "zipformer_s_55":
    from zipformer_s import ZipformerS, _load_zipformer_model
    from train_s import get_adjusted_batch_count, set_batch_count

if model_type == "zipformer_l_56":
    from zipformer_l import ZipformerL, _load_zipformer_model
    from train_l import get_adjusted_batch_count, set_batch_count

if model_type == "zipformer_m_55":
    from zipformer_m import ZipformerM, _load_zipformer_model
    from train_m import get_adjusted_batch_count, set_batch_count

from lhotse import CutSet, load_manifest
from lhotse.cut import MonoCut, Cut
from lhotse.features.io import NumpyHdf5Writer

from icefall.utils import AttributeDict, setup_logger
from icefall.lexicon import Lexicon
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler


class CodebookIndexExtractor:
    """
    A wrapper of quantiation.Quantizer.

    It's responsible for:
        1. extract and save activations from a teacher model.
        2. train quantizer from previous activations.
        3. extract codebook indexes for whole training set.
           Normally this step needs multi GPUs.
    """

    def __init__(self, params: AttributeDict):
        self.params = params
        params.subsets = ["train"]
        # if self.params.full_libri:
        #     self.params.subsets += ["clean-360", "other-500"]

        self.init_dirs()
        setup_logger(f"{self.vq_dir}/log-vq_extraction")

    def init_dirs(self):
        # vq_dir is the root dir for quantization, containing:
        # training data, trained quantizer, and extracted codebook indexes
        self.vq_dir = (
            self.params.kd_exp_dir
            / f"vq/{self.params.teacher_model_id}_layer{self.params.embedding_layer}_cb{self.params.num_codebooks}/"
        )
        self.vq_dir.mkdir(parents=True, exist_ok=True)

        # manifest_dir contains:
        # splited original manifests, extracted codebook indexes with related manifests # noqa
        self.manifest_dir = self.vq_dir / f"splits{self.params.world_size}"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        # It's doesn't matter whether ori_manifest_dir is str or Path.
        # Set it to Path to be consistent.
        self.ori_manifest_dir = Path("./data/fbank/")
        self.dst_manifest_dir = Path(
            # self.params.kd_exp_dir
            # / f"{self.params.embedding_layer}_cb{self.params.num_codebooks}/"
            self.vq_dir
        )

        self.dst_manifest_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        # Options about teacher embeddings eatraction.
        parser.add_argument(
            "--embedding-layer",
            type=int,
            help="layer to extract teacher embeddings, 1-based.",
            default=0,
        )

        parser.add_argument(
            "--num-utts",
            type=int,
            default=1000,
            help="num utts to train quantizer",
        )

        parser.add_argument(
            "--num-utts-extracted",
            type=int,
            default=30000,
            # default=360294,
            help="num utts to extract codebook index",
        )

        parser.add_argument(
            "--num-codebooks-teacher",
            type=int,
            default=8,
            help="""number of codebooks,
            i.e. number of codebook indexes each teacher embedding is compressed.
            """,
        )

    @property
    def embedding_file_path(self):
        """
        The saved embedding is used to train quantizer.
        """
        embedding_file_id = (
            f"num_utts_{self.params.num_utts}"
            + f"-layer_{self.params.embedding_layer}"
            + "-embedding_embeddings.h5"
        )

        embedding_file_path = self.vq_dir / embedding_file_id
        return embedding_file_path

    @torch.no_grad()
    def extract_and_save_embedding(self):
        """
        The extract embedding is used to train quantizer.
        """
        if self.embedding_file_path.exists():
            warn_message = (
                f"{self.embedding_file_path} already exists."
                + " Skip extracting embeddings from teacher model"
            )
            logging.warn(warn_message)
            return

        logging.info("Start to extract embeddings for training the quantizer.")
        total_cuts = 0
        with NumpyHdf5Writer(self.embedding_file_path) as writer:
            for batch_idx, batch in enumerate(self.quantizer_train_dl):
                cut_list = batch["supervisions"]["cut"]
                (
                    encoder_embedding,
                    num_frames,
                ) = self.teacher_model.extract_embedding(batch)
                # print("encoder_embedding shape : ", encoder_embedding.shape)
                encoder_embedding = encoder_embedding.cpu().numpy()
                for idx, cut in enumerate(cut_list):
                    cut.encoder_embedding = writer.store_array(
                        key=cut.id,
                        value=encoder_embedding[idx][: num_frames[idx]],
                    )
                    # print("encoder_embedding idx : ", encoder_embedding[idx].shape)
                    # print("num_frames idx : ", num_frames[idx])
                    # print("value : ", encoder_embedding[idx][: num_frames[idx]].shape)
                total_cuts += len(cut_list)
                logging.info(
                    f"Processed {total_cuts} output of {self.params.num_utts} cuts."
                )

        logging.info(f"Processed all {total_cuts} cuts.")

    @property
    def quantizer_train_dl(self):
        # used to train quantizer.
        aishell = AishellAsrDataModule(self.params)
        quantizer_trian_cuts = aishell.train_cuts().subset(
            first=self.params.num_utts
        )
        return aishell.train_dataloaders(quantizer_trian_cuts)

    @cached_property
    def quantizer_file_path(self):
        quantizer_file_id = (
            f"num_utts-{self.params.num_utts}"
            + f"-layer-{self.params.embedding_layer}"
            + f"-num_codebooks_{self.params.num_codebooks}"
            + "-quantizer.pt"
        )
        quantizer_file_path = Path(self.vq_dir) / quantizer_file_id

        return quantizer_file_path

    def train_quantizer(self):
        if self.quantizer_file_path.exists():
            warn_message = (
                f"{self.quantizer_file_path} already exists."
                + " Skip trainning quantizer."
            )
            logging.warn(warn_message)
            return

        assert self.embedding_file_path.exists()
        logging.info("Start to train quantizer.")
        trainer = quantization.QuantizerTrainer(
            dim=self.params.embedding_dim,
            bytes_per_frame=self.params.num_codebooks,
            device=self.params.device,
        )
        train, valid = quantization.read_hdf5_data(self.embedding_file_path)
        B = 512  # Minibatch size, this is very arbitrary,
        # it's close to what we used when we tuned this method.

        def minibatch_generator(data: torch.Tensor, repeat: bool):
            assert 3 * B < data.shape[0]
            cur_offset = 0
            while True if repeat else cur_offset + B <= data.shape[0]:
                start = cur_offset % (data.shape[0] + 1 - B)
                end = start + B
                cur_offset += B
                yield data[start:end, :].to(self.params.device).to(dtype=torch.float)

        for x in minibatch_generator(train, repeat=True):
            trainer.step(x)
            if trainer.done():
                break

        quantizer = trainer.get_quantizer()
        torch.save(quantizer.state_dict(), self.quantizer_file_path)

    def split_ori_manifests(self):
        """
        When multi gpus are available, split original manifests
        and extract codebook indexes in a prallel way.
        """
        for subset in self.params.subsets:
            logging.info(f"About to split {subset}.")
            ori_manifest = f"./data/fbank/librispeech_cuts_train-{subset}.jsonl.gz"
            split_cmd = f"lhotse split {self.params.world_size} {ori_manifest} {self.manifest_dir}"
            os.system(f"{split_cmd}")

    def join_manifests(self):
        """
        Join the vq manifest to the original manifest according to cut id.
        """
        logging.info("Start to join manifest files.")
        for subset in self.params.subsets:
            vq_manifest_path = (
                self.dst_manifest_dir / f"aishell_cuts_train-{subset}-vq.jsonl.gz"
            )
            ori_manifest_path = (
                self.ori_manifest_dir / f"aishell_cuts_{subset}.jsonl.gz"
            )
            dst_vq_manifest_path = (
                self.dst_manifest_dir / f"aishell_cuts_train-{subset}.jsonl.gz"
            )
            cuts_vq = load_manifest(vq_manifest_path)
            cuts_ori = load_manifest(ori_manifest_path)
            cuts_ori = cuts_ori.subset(first=self.params.num_utts_extracted)

            lexicon = Lexicon(self.params.lang_dir)
            graph_compiler = CharCtcTrainingGraphCompiler(
                lexicon=lexicon,
                device=self.params.device,
            )

            def remove_short_and_long_utt(c: Cut):
                if c.duration < 1.0 or c.duration > 12.0:
                    return False

                T = ((c.num_frames - 7) // 2 + 1) // 2
                tokens = graph_compiler.texts_to_ids([c.supervisions[0].text])[0]
                if T < len(tokens):
                    logging.warning(
                        f"Exclude cut with ID {c.id} from training. "
                        f"Number of frames (before subsampling): {c.num_frames}. "
                        f"Number of frames (after subsampling): {T}. "
                        f"Text: {c.supervisions[0].text}. "
                        f"Tokens: {tokens}. "
                        f"Number of tokens: {len(tokens)}"
                    )
                    return False

                return True

            cuts_ori = cuts_ori.filter(remove_short_and_long_utt)

            assert len(cuts_vq) == len(cuts_ori), "Cuts should have the same length!"

            if set(cuts_vq.ids) == set(cuts_ori.ids):
                # IDs match exactly
                cuts_vq = cuts_vq.sort_like(cuts_ori)
                for cut_idx, (cut_vq, cut_ori) in enumerate(zip(cuts_vq, cuts_ori)):
                    assert cut_vq.id == cut_ori.id, (cut_vq.id, cut_ori.id)
                    cut_ori.codebook_indexes = cut_vq.codebook_indexes
            else:
                # in case of ID mismatch, remap them
                # get the mapping between audio and cut ID
                logging
                ori_id_map = {}
                for id in cuts_ori.ids:
                    # some text normalization
                    if "sp" in id:
                        clean_id = "-".join(id.split("-")[:3]) + "_" + id.split("_")[-1]
                    else:
                        clean_id = "-".join(id.split("-")[:3])
                    ori_id_map[clean_id] = id

                for id in cuts_vq.ids:
                    if "sp" in id:
                        clean_id = "-".join(id.split("-")[:3]) + "_" + id.split("_")[-1]
                    else:
                        clean_id = "-".join(id.split("-")[:3])
                    assert clean_id in ori_id_map, clean_id
                    cuts_ori[ori_id_map[clean_id]].codebook_indexes = cuts_vq[
                        id
                    ].codebook_indexes

            CutSet.from_cuts(cuts_ori).to_jsonl(dst_vq_manifest_path)
            logging.info(f"Processed {subset}.")
            logging.info(f"Saved to {dst_vq_manifest_path}.")

    def merge_vq_manifests(self):
        """
        Merge generated vq included manfiests and storage to self.dst_manifest_dir.
        """
        for subset in self.params.subsets:
            vq_manifests = (
                f"{self.manifest_dir}/"
                + f"with_codebook_indexes-aishell-cuts_train-{subset}*.jsonl.gz"
            )
            dst_vq_manifest = (
                self.dst_manifest_dir / f"aishell_cuts_train-{subset}-vq.jsonl.gz"
            )
            if 1 == self.params.world_size:
                merge_cmd = f"cp {vq_manifests} {dst_vq_manifest}"
            else:
                merge_cmd = f"lhotse combine {vq_manifests} {dst_vq_manifest}"
            os.system(f"{merge_cmd}")

    def reuse_manifests(self):
        """
        Only train-* subsets are extracted codebook indexes from.
        The reset subsets are just a link from ./data/fbank.
        """

        def is_train(manifest: str) -> bool:
            for train_subset in ["train"]:
                if train_subset in manifest:
                    return True
            return False

        # Type of self.ori_nanifest_dir is Path
        # and result type of glob.glob is str.
        reusable_manifests = [
            manifest
            for manifest in glob.glob(f"{self.ori_manifest_dir}/*.gz")
            if not is_train(manifest)
        ]
        for manifest_path in reusable_manifests:
            ori_manifest_path = Path(manifest_path).resolve()
            # Path cannot used as a parameter of str.replace.
            # Cast them to str.
            dst_manifest_path = Path(
                manifest_path.replace(
                    str(self.ori_manifest_dir), str(self.dst_manifest_dir)
                )
            ).resolve()
            if not dst_manifest_path.exists():
                os.symlink(ori_manifest_path, dst_manifest_path)

    def create_vq_fbank(self):
        self.merge_vq_manifests()

    @cached_property
    def teacher_model(self):
        if model_type == "zipformer_l_56":
            return ZipformerL(self.params)
        if model_type == "zipformer_s_55":
            return ZipformerS(self.params)
        if model_type == "zipformer_m_55":
            return ZipformerM(self.params)

    @cached_property
    def quantizer(self):
        assert self.quantizer_file_path.exists()
        quantizer = quantization.Quantizer(
            dim=self.params.embedding_dim,
            num_codebooks=self.params.num_codebooks,
            codebook_size=256,
        )
        quantizer.load_state_dict(torch.load(self.quantizer_file_path))
        quantizer.to(self.params.device)
        return quantizer

    def load_ori_dl(self, subset):
        if self.params.world_size == 1:
            ori_manifest_path = f"./data/fbank/aishell_cuts_{subset}.jsonl.gz"
        else:
            ori_manifest_path = (
                self.manifest_dir
                / f"aishell_cuts_{subset}.{self.params.manifest_index}.jsonl.gz"  # noqa
            )
        aishell = AishellAsrDataModule(self.params)
        cuts = load_manifest(ori_manifest_path)
        train_cuts = cuts.subset(first=self.params.num_utts_extracted)  # type: CutSet

        lexicon = Lexicon(self.params.lang_dir)
        graph_compiler = CharCtcTrainingGraphCompiler(
            lexicon=lexicon,
            device=self.params.device,
        )

        def remove_short_and_long_utt(c: Cut):
            if c.duration < 1.0 or c.duration > 12.0:
                return False

            T = ((c.num_frames - 7) // 2 + 1) // 2
            tokens = graph_compiler.texts_to_ids([c.supervisions[0].text])[0]
            if T < len(tokens):
                logging.warning(
                    f"Exclude cut with ID {c.id} from training. "
                    f"Number of frames (before subsampling): {c.num_frames}. "
                    f"Number of frames (after subsampling): {T}. "
                    f"Text: {c.supervisions[0].text}. "
                    f"Tokens: {tokens}. "
                    f"Number of tokens: {len(tokens)}"
                )
                return False

            return True

        train_cuts = train_cuts.filter(remove_short_and_long_utt)
        dl = aishell.train_dataloaders(train_cuts)
        return dl

    def _release_gpu_memory(self):
        self.__dict__.pop("teacher_model", None)
        self.__dict__.pop("quantizer", None)
        torch.cuda.empty_cache()

    def extract_codebook_indexes(self):
        logging.info("Start to extract codebook indexes.")
        if self.params.world_size == 1:
            self.extract_codebook_indexes_imp()
        else:
            # Since a new extractor will be created for each rank in
            # compute_codebook_indexes_parallel, it's better to
            # release the GPU memory occupied by current extractor.
            self._release_gpu_memory()

            # Prepare split manifests for each job.
            self.split_ori_manifests()
            mp.spawn(
                compute_codebook_indexes_parallel,
                args=(self.params,),
                nprocs=self.params.world_size,
                join=True,
            )
        self.create_vq_fbank()

    @torch.no_grad()
    def extract_codebook_indexes_imp(self):
        for subset in self.params.subsets:
            num_cuts = 0
            new_cuts = []
            if self.params.world_size == 1:
                manifest_file_id = f"{subset}"
            else:
                manifest_file_id = f"{subset}-{self.params.manifest_index}"

            manifest_file_path = self.manifest_dir / manifest_file_id
            with NumpyHdf5Writer(manifest_file_path) as writer:
                for batch_idx, batch in enumerate(self.load_ori_dl(subset)):
                    if batch_idx % 10 == 0:
                        set_batch_count(model = self.teacher_model.model, batch_count = get_adjusted_batch_count(self.params))
                    (
                        encoder_embedding,
                        num_frames,
                    ) = self.teacher_model.extract_embedding(batch)
                    print("before quantizer embedding : ", encoder_embedding.shape)
                    codebook_indexes = self.quantizer.encode(encoder_embedding)
                    print("after quantizer codebook_indexes : ", codebook_indexes.shape)
                    # [N, T, C]
                    codebook_indexes = codebook_indexes.to("cpu").numpy()
                    assert np.min(codebook_indexes) >= 0
                    assert np.max(codebook_indexes) < 256
                    supervisions = batch["supervisions"]
                    cut_list = supervisions["cut"]
                    assert len(cut_list) == codebook_indexes.shape[0]
                    assert all(c.start == 0 for c in supervisions["cut"])

                    new_cut_list = []
                    for idx, cut in enumerate(cut_list):
                        new_cut = MonoCut(
                            id=cut.id,
                            start=cut.start,
                            duration=cut.duration,
                            channel=cut.channel,
                        )
                        new_cut.codebook_indexes = writer.store_array(
                            key=cut.id,
                            value=codebook_indexes[idx][: num_frames[idx]],
                            frame_shift=0.02,
                            temporal_dim=0,
                            start=0,
                        )
                        new_cut_list.append(new_cut)
                    new_cuts += new_cut_list
                    num_cuts += len(cut_list)
                    message = f"Processed {num_cuts} cuts from {subset}"
                    if self.params.world_size > 1:
                        message += f" by job {self.params.manifest_index}"
                    logging.info(f"{message}.")

                json_file_path = (
                    self.manifest_dir
                    / f"with_codebook_indexes-aishell-cuts_train-{manifest_file_id}.jsonl.gz"  # noqa
                )
                CutSet.from_cuts(new_cuts).to_jsonl(json_file_path)


@torch.no_grad()
def compute_codebook_indexes_parallel(
    rank: int,
    params,
) -> List[Tuple[str, List[int]]]:
    """Create an extractor for each rank and extract codebook indexes parallelly.

    Normally, this function is called by torch.multiprocessing
    when multi GPUs are available.
    """
    params = copy.deepcopy(params)
    device = torch.device("cuda", rank)
    params.device = device

    # rank is 0-based while split manifests by "lhotse split" is 1-based.
    params.manifest_index = rank + 1

    extractor = CodebookIndexExtractor(params=params)
    extractor.extract_codebook_indexes_imp()
