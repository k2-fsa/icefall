import collections
import logging
import os
import re
from typing import List, Tuple

import torch
import torch.distributed as dist
from lhotse.array import Array, TemporalArray
from torch.utils.tensorboard import SummaryWriter

from icefall.byte_utils import byte_encode
from icefall.utils import tokenize_by_CJK_char


def _normalize_chinese_text(text):
    # 去除所有标点符号
    text = re.sub(r"[，。！？、；：“”‘’（）《》【】{}·…—～]", "", text)
    # 去除汉字之间的空格（确保不影响英文单词）
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = text.upper()
    return text


def normalize_chinese_text(c):
    text = c.supervisions[0].text
    text = _normalize_chinese_text(text)
    c.supervisions[0].text = text
    return c


def _normalize_english_text(text):
    # 只保留字母、数字、空格和单引号，去掉其他标点符号
    text = re.sub(r"[^\w\s']", "", text)
    # 转换为大写
    text = text.upper()
    return text


def normalize_english_text(c):
    text = c.supervisions[0].text
    text = _normalize_english_text(text)
    c.supervisions[0].text = text
    return c


def remove_non_alphabetic(text: str, strict: bool = True) -> str:
    # Recommend to set strict to False
    if not strict:
        # Note, this also keeps space, single quote(')
        text = text.replace("-", " ")
        text = text.replace("—", " ")
        return re.sub(r"[^a-zA-Z0-9\s']+", "", text)
    else:
        # only keeps space
        return re.sub(r"[^a-zA-Z\s]+", "", text)


def map_zh(c):
    text = c.supervisions[0].text
    text = byte_encode(tokenize_by_CJK_char(text))
    c.supervisions[0].text = text
    return c


def upper_only_alpha(c):
    text = c.supervisions[0].text
    text = remove_non_alphabetic(text.upper(), strict=False)
    c.supervisions[0].text = text
    return c


def add_dummy_text(c):
    if c.supervisions[0].text is None:
        c.supervisions[
            0
        ].text = "Dummy text added as a place holder. Please ignore this if possible."
    return c


def _add_dummy_embeddings_and_taskIDs(task_ID: int, c):
    whisper_embedding_dict = {
        "array": {
            "storage_type": "numpy_hdf5",
            "storage_path": "data/dummy_embeddings/dummy_whisper_embedding_1510.h5",
            "storage_key": "dummy_whisper_embedding_1510",
            "shape": [1510, 1280],
        },
        "temporal_dim": 0,
        "frame_shift": 0.02,
        "start": 0,
    }
    whisper_dummy_embedding = TemporalArray.from_dict(whisper_embedding_dict)

    whisper_cb_indexes_dict = {
        "array": {
            "storage_type": "numpy_hdf5",
            "storage_path": "data/dummy_embeddings/dummy_whisper_codebook_indexes_1510.h5",
            "storage_key": "dummy_whisper_codebook_indexes_1510",
            "shape": [1510, 16],
        },
        "temporal_dim": 0,
        "frame_shift": 0.02,
        "start": 0,
    }
    whisper_cb_indexes = TemporalArray.from_dict(whisper_cb_indexes_dict)

    beats_embedding_dict = {
        "storage_type": "numpy_hdf5",
        "storage_path": "data/dummy_embeddings/dummy_beats_embedding.h5",
        "storage_key": "dummy_beats_embedding",
        "shape": [527],
    }
    beats_dummy_embedding = Array.from_dict(beats_embedding_dict)

    ecapa_embedding_dict = {
        "storage_type": "numpy_hdf5",
        "storage_path": "dummy_ecapa_embedding.h5",
        "storage_key": "dummy_ecapa_embedding",
        "shape": [1, 192],
    }
    ecapa_dummy_embedding = Array.from_dict(ecapa_embedding_dict)

    mert_embedding_dict = {
        "array": {
            "storage_type": "numpy_hdf5",
            "storage_path": "data/dummy_embeddings/dummy_mert_embedding_2260.h5",
            "storage_key": "dummy_mert_embedding",
            "shape": [2260, 1024],
        },
        "temporal_dim": 0,
        "frame_shift": 0.013333333333333334,
        "start": 0,
    }
    mert_dummy_embedding = TemporalArray.from_dict(mert_embedding_dict)

    def add_embeddings(c):
        # if not c.has_custom("whisper_embedding"):
        #     c.whisper_embedding = whisper_dummy_embedding
        if not c.has_custom("codebook_indexes"):
            c.codebook_indexes = whisper_cb_indexes

        # if not c.has_custom("ecapa_embedding"):
        #     c.ecapa_embedding = ecapa_dummy_embedding
        if not c.has_custom("beats_embedding"):
            c.beats_embedding = beats_dummy_embedding
        # if not c.supervisions[0].has_custom("audio_event"):
        #     c.supervisions[0].audio_event = "0"
        if c.supervisions[0].text is None:
            c.supervisions[
                0
            ].text = (
                "Dummy text added as a place holder. Please ignore this if possible."
            )
        if task_ID is not None:
            c.task_id = task_ID
        return c

    c = add_embeddings(c)
    return c


def _add_task_id(task_id, c):
    c.task_id = task_id
    return c


def _add_language_id(lid, c):
    c.language_id = lid
    return c



def _save_checkpoint_with_global_batch_idx(
    params,
    model,
    optimizer=None,
    sampler=None,
    scheduler=None,
    scaler=None,
    model_avg=None,
    rank: int = 0,
):
    # only active when rank==0
    if rank != 0:
        return

    if isinstance(model, DDP):
        model = model.module
    else:
        model = model

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }

    if model_avg is not None:
        checkpoint["model_avg"] = model_avg.to(torch.float32).state_dict()

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v
    output_path = params.exp_dir / f"checkpoint-{params.batch_idx_train}.pt"

    if params.save_with_client:
        output_path = "brainllm:s3://yangxiaoyu/" + str(output_path)
        logging.info(f"Saving checkpoint to {output_path}")
        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            f.seek(0)
            params.client.put(output_path, f)
        logging.info(f"Finish saving checkpoint to {output_path}")
    else:
        logging.info(f"Saving checkpoint to {output_path}")
        torch.save(checkpoint, output_path)


def _save_checkpoint(
    filename,
    model,
    model_avg=None,
    params=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    sampler=None,
    rank: int = 0,
):
    if rank != 0:
        return

    logging.info(f"Saving checkpoint to {filename}")

    if isinstance(model, DDP):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
    }

    if model_avg is not None:
        checkpoint["model_avg"] = model_avg.to(torch.float32).state_dict()

    if params:
        for k, v in params.items():
            assert k not in checkpoint
            checkpoint[k] = v

    if "s3://" in filename:
        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            f.seek(0)
            params.client.put(filename, f)
        logging.info(f"Finish saving checkpoint to {filename}")
    else:
        torch.save(checkpoint, filename)


class MetricsTracker(collections.defaultdict):
    def __init__(self, normalize: bool = True):
        # Passing the type 'int' to the base-class constructor
        # makes undefined items default to int() which is zero.
        # This class will play a role as metrics tracker.
        # It can record many metrics, including but not limited to loss.
        super(MetricsTracker, self).__init__(int)
        self.normalize = normalize

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            if v - v == 0:
                ans[k] = ans[k] + v
        return ans

    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans

    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."

        return ans_frames + ans_utterances

    def norm_items(self) -> List[Tuple[str, float]]:
        """
        Returns a list of pairs, like:
          [('ctc_loss', 0.1), ('att_loss', 0.07)]
        """
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            if not self.normalize:
                ans.append((k, float(v)))
                continue
            if ("audio_tagging" in k) or ("speaker_verification" in k):
                norm_value = float(v) / num_utterances
            else:
                norm_value = (
                    float(v) / num_frames
                    if "utt_" not in k
                    else float(v) / num_utterances
                )
            ans.append((k, norm_value))
        return ans

    def reduce(self, device):
        """
        Reduce using torch.distributed, which I believe ensures that
        all processes get the total.
        """
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v

    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        """Add logging information to a TensorBoard writer.

        Args:
            tb_writer: a TensorBoard writer
            prefix: a prefix for the name of the loss, e.g. "train/valid_",
                or "train/current_"
            batch_idx: The current batch index, used as the x-axis of the plot.
        """
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)


if __name__ == "__main__":
    text = "你好 ， 这是 一个  测试 句子 ！Hello 希望 这段 代码 能正常 工作 。"
    normalized_text = normalize_chinese_text(text)
    print(normalized_text)

    text = "Hello, world! It's a great day to learn NLP."
    normalized_text = normalize_english_text(text)
    print(normalized_text)
