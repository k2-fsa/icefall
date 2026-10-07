#!/usr/bin/env python3
# Copyright    2026  (authors: Ruizhe Huang, Mahsa Yarmohammadi)
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


"""
A light-weight CPU test of the contextual biasing model. It needs no data:
a tiny BPE model and fake biasing word lists are created in a temp dir.

To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./pruned_transducer_stateless7_contextual/test_model.py
"""

import contextlib
import io
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
from context_collector import ContextCollector
from decode import decode_one_batch
from decode import get_params as get_decode_params
from decode import get_parser as get_decode_parser
from icefall.checkpoint import average_checkpoints_with_averaged_model
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from score import main as score_main
from train import (
    compute_loss,
    get_params,
    get_parser,
    get_transducer_model,
    get_word_encoder,
    load_pretrained_asr,
)

# A tiny Zipformer so that the test runs in seconds on CPU
TINY_MODEL_ARGS = [
    "--num-encoder-layers", "1,1,1,1,1",
    "--feedforward-dims", "64,64,64,64,64",
    "--nhead", "2,2,2,2,2",
    "--encoder-dims", "32,32,32,32,32",
    "--attention-dims", "16,16,16,16,16",
    "--encoder-unmasked-dims", "16,16,16,16,16",
    "--decoder-dim", "32",
    "--joiner-dim", "32",
]  # fmt: skip


def make_fixtures(d: Path):
    random.seed(0)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def word():
        return "".join(random.choices(letters, k=random.randint(3, 8)))

    common = sorted({word() for _ in range(100)})
    rare = sorted({word() for _ in range(500)} - set(common))
    (d / "ctx/words").mkdir(parents=True)
    (d / "ctx/words/common_words_5k.txt").write_text("\n".join(common) + "\n")
    (d / "ctx/words/all_rare_words.txt").write_text("\n".join(rare) + "\n")

    text = [" ".join(random.choices(common + rare, k=10)) for _ in range(500)]
    (d / "text.txt").write_text("\n".join(text) + "\n")
    spm.SentencePieceTrainer.train(
        input=str(d / "text.txt"),
        model_prefix=str(d / "bpe"),
        vocab_size=100,
        model_type="unigram",
        character_coverage=1.0,
        unk_id=2,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=["<blk>", "<sos/eos>"],
        minloglevel=2,
    )
    return common, rare


def setup(d: Path):
    common, rare = make_fixtures(d)
    argv = (
        TINY_MODEL_ARGS
        + ["--bpe-model", str(d / "bpe.model"), "--context-dir", str(d / "ctx")]
        + ["--n-distractors", "10"]
    )
    params = get_params()
    params.update(get_decode_params())
    params.update(vars(get_decode_parser().parse_known_args(argv)[0]))
    params.update(vars(get_parser().parse_args(argv)))
    params.max_duration = 100
    params.biased_lm_scale = 0.5

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    params.backoff_id = params.vocab_size

    context_collector = ContextCollector(
        path_is21_deep_bias=Path(params.context_dir),
        sp=sp,
        n_distractors=params.n_distractors,
        backoff_id=params.backoff_id,
    )
    model = get_transducer_model(params)
    model.params = params

    texts = [
        " ".join([common[0], rare[3], common[5], rare[7]]),
        " ".join([common[1], rare[9]]),
    ]
    num_frames = torch.tensor([100, 80])
    batch = {
        "inputs": torch.randn(2, 100, 80),
        "supervisions": {"text": texts, "num_frames": num_frames},
    }
    return params, sp, context_collector, model, batch, common, rare


def test_train_step(params, sp, context_collector, model, batch):
    for p in model.parameters():
        p.requires_grad = False
    for m in (
        model.context_encoder,
        model.encoder_biasing_adapter,
        model.decoder_biasing_adapter,
    ):
        for p in m.parameters():
            p.requires_grad = True

    model.train()
    loss, info = compute_loss(
        params, model, context_collector, sp, batch, is_training=True
    )
    loss.backward()
    assert torch.isfinite(loss), info

    biasing = ("context_encoder", "encoder_biasing_adapter", "decoder_biasing_adapter")
    with_grad = {n.split(".")[0] for n, p in model.named_parameters() if p.grad is not None}
    assert with_grad == set(biasing), with_grad
    print(f"train step OK: {info}")


def test_init_asr_ckpt(params, model, d: Path):
    """--init-asr-ckpt: load a checkpoint that has no biasing modules."""
    biasing = ("context_encoder.", "encoder_biasing_adapter.", "decoder_biasing_adapter.")
    asr_state = {k: v for k, v in model.state_dict().items() if not k.startswith(biasing)}
    torch.save({"model": asr_state}, d / "asr.pt")

    new_model = get_transducer_model(params)
    load_pretrained_asr(str(d / "asr.pt"), new_model)
    for k, v in asr_state.items():
        assert torch.equal(new_model.state_dict()[k], v), k
    print("init-asr-ckpt OK")


def test_predefined_lists_and_scoring(sp, common, rare, d: Path):
    """Predefined biasing lists (--is-predefined) and U-WER/B-WER scoring,
    using a tiny file in the format of fbai-speech/is21_deep_bias/ref."""
    utts = {
        "1-1-0001": ([common[0], rare[3], common[5]], [rare[3]], rare[3:8]),
        "1-1-0002": ([rare[9], common[1]], [rare[9]], rare[8:12]),
    }
    (d / "ctx/ref").mkdir()
    for name in ("test-clean", "test-other"):
        with open(d / f"ctx/ref/{name}.biasing_100.tsv", "w") as f:
            for uid, (text, biased, context) in utts.items():
                fields = [" ".join(text), json.dumps(biased), json.dumps(context)]
                print(uid, *[x.lower() for x in fields], sep="\t", file=f)

    collector = ContextCollector(
        path_is21_deep_bias=d / "ctx", sp=sp, is_predefined=True, n_distractors=100
    )
    cuts = [SimpleNamespace(supervisions=[SimpleNamespace(id=u)]) for u in utts]
    batch = {"supervisions": {"cut": cuts}}
    _, _, num_words_per_utt = collector.get_context_word_list(batch)
    assert num_words_per_utt == [len(c) for _, _, c in utts.values()], num_words_per_utt

    # The 2nd hypothesis misses its only biased word: B-WER = 50%, U-WER = 0%
    hyps = {
        "1-1-0001": " ".join(utts["1-1-0001"][0]).lower(),
        "1-1-0002": common[1].lower(),
    }
    args = SimpleNamespace(refs=d / "ctx/ref/test-clean.biasing_100.tsv", hyps=hyps, lenient=True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        score_main(args)
    summary = out.getvalue().strip().splitlines()[-1]
    assert summary == "20.00(0.00/50.00)", summary
    print(f"predefined lists + scoring OK: WER(U-WER/B-WER) = {summary}")


def test_average_checkpoints(params, model, d: Path):
    """Checkpoints hold params (e.g., exp_dir as a PosixPath), which
    torch>=2.6 refuses to load with weights_only=True; decode.py averages
    them with --use-averaged-model true."""
    params.exp_dir = d
    for epoch in (1, 2):
        params.batch_idx_train = 200 * epoch
        save_checkpoint_impl(
            filename=d / f"epoch-{epoch}.pt", model=model, model_avg=model, params=params
        )
    avg = average_checkpoints_with_averaged_model(
        filename_start=str(d / "epoch-1.pt"), filename_end=str(d / "epoch-2.pt"), device="cpu"
    )
    assert avg.keys() == model.state_dict().keys()
    print("checkpoint averaging OK")


def test_asr_eval_mode_and_context_dim(params):
    """--asr-eval-mode keeps the frozen ASR in eval mode through model.train();
    --context-dim sets the size of the biasing modules."""
    model = get_transducer_model(params)
    model.asr_eval_mode = True
    model.train()
    assert not model.encoder.training and not model.decoder.training
    assert not model.joiner.training
    assert model.context_encoder.training and model.encoder_biasing_adapter.training
    model.asr_eval_mode = False
    model.train()
    assert model.encoder.training

    sizes = {}
    for dim in (params.context_dim, 2 * params.context_dim):
        params.context_dim, saved = dim, params.context_dim
        m = get_transducer_model(params)
        params.context_dim = saved
        assert m.encoder_biasing_adapter.proj_in1.out_features == dim
        sizes[dim] = sum(p.numel() for p in m.encoder_biasing_adapter.parameters())
    assert sizes[2 * params.context_dim] > sizes[params.context_dim], sizes
    print("asr-eval-mode and context-dim OK")


def test_pretrained_word_encoder(params, common, rare, batch, d: Path):
    """--is-pretrained-context-encoder with fastText-style embeddings:
    the same word encoder must be used for training and decoding."""
    with open(d / "embeddings.txt", "w") as f:
        for w in common + rare:
            print(w.lower(), *[f"{x:.3f}" for x in torch.randn(300).tolist()], file=f)

    params = type(params)(dict(params))
    params.is_pretrained_context_encoder = True
    params.pretrained_word_encoder = "fasttext"
    params.fasttext_embeddings = str(d / "embeddings.txt")
    params.fasttext_model = str(d / "not-needed.bin")  # all words are in the file
    word_encoder = get_word_encoder(params, torch.device("cpu"))
    assert params.context_embedding_size == 300

    collector = ContextCollector(
        path_is21_deep_bias=Path(params.context_dir),
        sp=None,
        bert_encoder=word_encoder,
        n_distractors=params.n_distractors,
        backoff_id=params.backoff_id,
    )
    model = get_transducer_model(params)
    model.params = params
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    model.train()
    loss, _ = compute_loss(params, model, collector, sp, batch, is_training=True)
    loss.backward()
    assert torch.isfinite(loss)

    model.eval()
    params.decoding_method = "modified_beam_search"
    params.beam_size = 2
    model.no_encoder_biasing = params.no_encoder_biasing = False
    model.no_decoder_biasing = params.no_decoder_biasing = False
    model.no_wfst_lm_biasing = params.no_wfst_lm_biasing = True
    with torch.no_grad():
        hyps = decode_one_batch(params, model, collector, sp, batch)
    assert len(next(iter(hyps.values()))) == 2
    print("pretrained word encoder (fastText) train + decode OK")


def test_decode(params, sp, context_collector, model, batch):
    model.eval()
    # (method, encoder biasing, decoder biasing, WFST biasing)
    configs = [
        ("greedy_search", True, False, False),
        ("greedy_search", False, False, False),
        ("modified_beam_search", True, True, False),
        ("modified_beam_search", True, True, True),
        ("modified_beam_search", False, False, False),
    ]
    for method, enc, dec, wfst in configs:
        params.decoding_method = method
        params.beam_size = 2
        model.no_encoder_biasing = params.no_encoder_biasing = not enc
        model.no_decoder_biasing = params.no_decoder_biasing = not dec
        model.no_wfst_lm_biasing = params.no_wfst_lm_biasing = not wfst
        with torch.no_grad():
            hyps = decode_one_batch(params, model, context_collector, sp, batch)
        (key,) = hyps.keys()
        assert len(hyps[key]) == 2, hyps
        print(f"decode OK: {method}, encoder={enc}, decoder={dec}, wfst={wfst}")

    # greedy_search does not implement decoder-side biasing
    params.decoding_method = "greedy_search"
    model.no_decoder_biasing = params.no_decoder_biasing = False
    try:
        decode_one_batch(params, model, context_collector, sp, batch)
    except AssertionError:
        print("decode OK: greedy_search rejects decoder biasing")
    else:
        raise AssertionError("greedy_search should reject decoder biasing")


def main():
    torch.manual_seed(20260922)
    with tempfile.TemporaryDirectory() as d:
        params, sp, context_collector, model, batch, common, rare = setup(Path(d))
        test_predefined_lists_and_scoring(sp, common, rare, Path(d))
        test_init_asr_ckpt(params, model, Path(d))
        test_asr_eval_mode_and_context_dim(params)
        test_train_step(params, sp, context_collector, model, batch)
        test_average_checkpoints(params, model, Path(d))
        test_pretrained_word_encoder(params, common, rare, batch, Path(d))
        test_decode(params, sp, context_collector, model, batch)


if __name__ == "__main__":
    main()
