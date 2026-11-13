#!/usr/bin/env python3
"""CPU checks for LibriSpeech data, CE packing and checkpoint compatibility."""

import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from asr_datamodule import LibriSpeechAsrDataModule
from decode import configure_params, get_parser, merge_partials, prepare_checkpoint
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from model import SPEECH_LLM
from prepare import prepare_librispeech, prepare_musan
from resume_ce import load_trainable, resume_tag
from train import get_deepspeed_config
from train import get_parser as train_parser

from icefall.utils import AttributeDict


class RecipeTests(unittest.TestCase):
    def test_size_profiles_preserve_the_step_schedule(self):
        for size, duration, accumulation, lr, checkpointing, bucket in [
            ("0.6B", 200, 1, 1e-4, False, 200000000),
            ("1.7B", 200, 1, 1e-4, False, 200000000),
            ("4B", 200, 1, 1e-4, False, 200000000),
            ("8B", 200, 1, 1e-4, False, 200000000),
            ("14B", 100, 2, 1e-4, False, 200000000),
            ("32B", 100, 2, 5e-5, True, 50000000),
        ]:
            with self.subTest(size=size):
                parser = train_parser()
                LibriSpeechAsrDataModule.add_arguments(parser)
                args = parser.parse_args(
                    [
                        "--max-duration",
                        str(duration),
                        "--gradient-accumulation-steps",
                        str(accumulation),
                        "--lr",
                        str(lr),
                        "--activation-checkpointing",
                        str(checkpointing),
                        "--ds-bucket-size",
                        str(bucket),
                        "--stop-after-steps",
                        "2",
                    ]
                )
                config = get_deepspeed_config(AttributeDict(vars(args)))
                self.assertEqual(
                    config["scheduler"]["params"]["total_num_steps"], 50000
                )
                self.assertEqual(config["optimizer"]["params"]["lr"], lr)
                self.assertEqual(config["gradient_accumulation_steps"], accumulation)
                self.assertEqual(duration * accumulation * 8, 1600)
                self.assertTrue(args.use_lora)
                self.assertFalse(args.unfreeze_encoder)

    def test_standard_cuts_keep_eval_and_generate_128_bin_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "manifests"
            output = root / "fbank"
            manifests.mkdir()
            output.mkdir()
            wav = root / "speech.wav"
            sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
            for split in [
                "train-clean-100",
                "train-clean-360",
                "train-other-500",
                "test-clean",
            ]:
                recording = Recording.from_file(wav, recording_id=split)
                RecordingSet.from_recordings([recording]).to_file(
                    manifests / f"librispeech_recordings_{split}.jsonl.gz"
                )
                SupervisionSet.from_segments(
                    [
                        SupervisionSegment(
                            id=split,
                            recording_id=split,
                            start=0,
                            duration=1,
                            text="HELLO WORLD",
                        )
                    ]
                ).to_file(manifests / f"librispeech_supervisions_{split}.jsonl.gz")
            splits = [
                "train-clean-100",
                "train-clean-360",
                "train-other-500",
                "test-clean",
            ]
            prepare_librispeech(manifests, output, splits, perturb_speed=True)
            train = CutSet.from_file(
                output / "librispeech_cuts_train-clean-100.jsonl.gz"
            )
            self.assertEqual(len(train), 3)
            evaluation = CutSet.from_file(
                output / "librispeech_cuts_test-clean.jsonl.gz"
            )
            self.assertEqual(len(evaluation), 1)
            self.assertEqual(next(iter(evaluation)).duration, 1)
            noise = root / "noise.wav"
            sf.write(noise, np.zeros(12 * 16000, dtype=np.float32), 16000)
            for part in ["music", "speech", "noise"]:
                RecordingSet.from_recordings(
                    [Recording.from_file(noise, recording_id=part)]
                ).to_file(manifests / f"musan_recordings_{part}.jsonl.gz")
            prepare_musan(manifests, output)
            self.assertEqual(len(CutSet.from_file(output / "musan_cuts.jsonl.gz")), 3)
            parser = argparse.ArgumentParser()
            LibriSpeechAsrDataModule.add_arguments(parser)
            args = parser.parse_args(
                ["--manifest-dir", str(output), "--num-workers", "0"]
            )
            module = LibriSpeechAsrDataModule(args)
            self.assertEqual(len(list(module.train_cuts())), 9)
            repeated = CutSet.from_cuts(
                [replace(next(iter(evaluation)), id=f"test-{i}") for i in range(12)]
            )
            batch = next(iter(module.test_dataloaders(repeated)))
            self.assertEqual(batch["inputs"].shape[-1], 128)
            self.assertEqual(len(batch["supervisions"]["cut"]), 12)

    def test_checkpoint_average_and_split_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            for step, value in [(2000, 1.0), (4000, 3.0)]:
                folder = exp / f"checkpoint-{step}"
                folder.mkdir()
                torch.save(
                    {"weight": torch.tensor([value])}, folder / "pytorch_model.bin"
                )
            args = get_parser().parse_args(
                [
                    "--exp-dir",
                    str(exp),
                    "--step",
                    "4000",
                    "--avg",
                    "2",
                    "--dataset",
                    "test-clean",
                    "test-other",
                    "--beam-size",
                    "4",
                ]
            )
            params = configure_params(args)
            path, temporary = prepare_checkpoint(params)
            self.assertTrue(temporary)
            self.assertEqual(torch.load(path, weights_only=True)["weight"].item(), 2)
            self.assertEqual(params.dataset, ["test-clean", "test-other"])
            for split in params.dataset:
                for rank in range(2):
                    ids = (
                        [f"{split}-{rank}"]
                        if rank == 0
                        else [f"{split}-1", f"{split}-0_dup0"]
                    )
                    torch.save(
                        {"beam-search": [(i, ["HELLO"], ["HELLO"]) for i in ids]},
                        params.res_dir / f".partial-{split}-{rank}.pt",
                    )
                merged = merge_partials(params, 2, split)
                self.assertEqual(len(merged["beam-search"]), 2)

    def test_resume_rejects_missing_trainable_weights(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaises(RuntimeError):
            load_trainable({"weight": model.weight.detach()}, model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(resume_tag(root, True))
            (root / "latest").write_text("checkpoint-2\n")
            with self.assertRaises(RuntimeError):
                resume_tag(root, True)
            (root / "checkpoint-2").mkdir()
            (root / "checkpoint-2/ce_recipe.json").write_text("{}")
            torch.save({}, root / "checkpoint-2-sampler.pt")
            self.assertEqual(resume_tag(root, True), "checkpoint-2")

    def test_speech_padding_uses_true_lengths(self):
        llm = torch.nn.Module()
        llm.config = SimpleNamespace(default_speech_token_id=9)
        model = SPEECH_LLM(torch.nn.Identity(), llm, torch.nn.Identity())
        ids = torch.tensor([[1, 9, 2, 3], [1, 9, 2, 0]])
        mask = ids.ne(0)
        embeddings = torch.ones(2, 4, 2)
        speech = torch.full((2, 3, 2), 7.0)
        labels = torch.tensor([[-100, -100, 2, 3], [-100, -100, 2, -100]])
        for side in ["left", "right"]:
            (
                merged,
                merged_mask,
                targets,
                positions,
            ) = model._merge_input_ids_with_speech_features(
                speech,
                embeddings,
                ids,
                mask,
                labels,
                speech_lens=torch.tensor([3, 1]),
                padding_side=side,
            )
            self.assertEqual(merged_mask.sum(dim=1).tolist(), [6, 3])
            self.assertEqual((targets != -100).sum(dim=1).tolist(), [2, 1])
            self.assertEqual(
                positions[merged_mask.bool()].tolist(), [0, 1, 2, 3, 4, 5, 0, 1, 2]
            )


if __name__ == "__main__":
    unittest.main()
