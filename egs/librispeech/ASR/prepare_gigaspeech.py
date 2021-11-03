#!/usr/bin/env python3

# Copyright (c)  2021  Johns Hopkins University (Piotr Żelasko)
# Apache 2.0
import argparse
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from gigaspeech_datamodule import get_context_suffix
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomHdf5Writer,
    SupervisionSegment,
    combine,
)
from lhotse.recipes import prepare_gigaspeech, prepare_musan
from lhotse.utils import is_module_available
from icefall.utils import str2bool

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@contextmanager
def get_executor():
    # We'll either return a process pool or a distributed worker pool.
    # Note that this has to be a context manager because we might use multiple
    # context manager ("with" clauses) inside, and this way everything will
    # free up the resources at the right time.
    try:
        # If this is executed on the CLSP grid, we will try to use the
        # Grid Engine to distribute the tasks.
        # Other clusters can also benefit from that, provided a cluster-specific wrapper.
        # (see https://github.com/pzelasko/plz for reference)
        #
        # The following must be installed:
        # $ pip install dask distributed
        # $ pip install git+https://github.com/pzelasko/plz
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster() as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except:
        pass
    # No need to return anything - compute_and_store_features
    # will just instantiate the pool itself.
    yield None


def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    print(
        "Please create a place on your system to put the downloaded Librispeech data "
        "and add it to `corpus_dirs`"
    )
    sys.exit(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=min(5, os.cpu_count()),
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="XS",
        help="Select the GigaSpeech subset (XS|S|M|L|XL)",
    )
    parser.add_argument(
        "--context-window",
        type=float,
        default=0.0,
        help="Training cut duration in seconds. "
        "Use 0 to train on supervision segments without acoustic context, with variable cut lengths; "
        "number larger than zero will create multi-supervisions cuts with actual acoustic context. ",
    )
    parser.add_argument(
        "--context-direction",
        type=str,
        default="center",
        help="If context-window is 0, does nothing. "
        "If it's larger than 0, determines in which direction (relative to the supervision) "
        "to seek for extra acoustic context. Available values: (left|right|center|random).",
    )
    parser.add_argument(
        "--precomputed-features",
        type=str2bool,
        default=True,
        help="Should we pre-compute features and store them on disk or not. "
        "It is recommended to disable it for L and XL splits as the pre-computation "
        "might currently consume excessive memory and time -- use on-the-fly feature "
        "extraction in the training script instead.",
    )
    return parser


# Similar text filtering and normalization procedure as in:
# https://github.com/SpeechColab/GigaSpeech/blob/main/toolkits/kaldi/gigaspeech_data_prep.sh


def normalize_text(
    utt: str,
    punct_pattern=re.compile(r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>"),
    whitespace_pattern=re.compile(r"\s\s+"),
) -> str:
    return whitespace_pattern.sub(" ", punct_pattern.sub("", utt))


def has_no_oov(
    sup: SupervisionSegment, oov_pattern=re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>")
) -> bool:
    return oov_pattern.search(sup.text) is None


def main():
    args = get_parser().parse_args()
    dataset_parts = [args.subset, "DEV", "TEST"]
    if args.subset in ["L", "XL"]:
        assert is_module_available("pyarrow"), (
            "Running the GigaSpeech recipe for L and XL splits "
            "currently requires installing optional dependencies: "
            "'pip install pyarrow pandas'."
        )

    print("Parts we will prepare: ", dataset_parts)

    corpus_dir = locate_corpus(
        Path("/export/corpora5/gigaspeech"),
        Path("/exp/pzelasko/gigaspeech"),
        Path("/home/storage07/zhangjunbo/data/GigaSpeech")
    )
    musan_dir = locate_corpus(
        Path("/export/corpora5/JHU/musan"),
        Path("/export/common/data/corpora/MUSAN/musan"),
        Path("/root/fangjun/data/musan"),
    )

    output_dir = Path("exp/data")
    print("GigaSpeech manifest preparation:")
    gigaspeech_manifests = prepare_gigaspeech(
        corpus_dir=corpus_dir,
        dataset_parts=dataset_parts,
        output_dir=output_dir,
        num_jobs=args.num_jobs,
    )

    print("Musan manifest preparation:")
    musan_cuts_path = output_dir / "cuts_musan.json.gz"
    musan_manifests = prepare_musan(
        corpus_dir=musan_dir, output_dir=output_dir, parts=("music", "speech", "noise")
    )

    ctx_suffix = get_context_suffix(args)

    print("Feature extraction:")
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    with get_executor() as ex:  # Initialize the executor only once.
        for partition, manifests in gigaspeech_manifests.items():
            # For L and XL partition we are going to store the manifest using pyarrow.
            cuts_path_ext = "jsonl.gz" if partition not in ["L", "XL"] else "arrow"
            raw_cuts_path = output_dir / f"gigaspeech_cuts_{partition}_raw.jsonl.gz"
            cuts_path = (
                output_dir / f"gigaspeech_cuts_{partition}{ctx_suffix}.{cuts_path_ext}"
            )

            if raw_cuts_path.is_file():
                print(f"{partition} already exists - skipping checking transcript.")
            else:
                # Note this step makes the recipe different than LibriSpeech:
                # We must filter out some utterances and remove punctuation to be consistent with Kaldi.
                print("Filtering OOV utterances from supervisions")
                manifests["supervisions"] = manifests["supervisions"].filter(has_no_oov)
                print("Normalizing text in", partition)
                for sup in manifests["supervisions"]:
                    sup.text = normalize_text(sup.text)

                # Create long-recording cut manifests.
                print("Processing", partition)
                cut_set = CutSet.from_manifests(
                    recordings=manifests["recordings"],
                    supervisions=manifests["supervisions"],
                )

                # Run data augmentation that needs to be done in the time domain.
                if partition not in ["DEV", "TEST"]:
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )

                cut_set.to_file(raw_cuts_path)

            if cuts_path.is_file():
                print(
                    f"{partition} already exists - skipping cutting into sub-segments and feature extraction."
                )
            else:
                try:
                    # If we skipped initializing `cut_set` because it exists on disk, we'll load it.
                    # This helps us avoid re-computing the features for different variants of
                    # context windows.
                    cut_set
                except NameError:
                    print(f"Reading {partition} raw cuts from disk.")
                    cut_set = CutSet.from_file(raw_cuts_path)
                # Note this step makes the recipe different than LibriSpeech:
                # Since recordings are long, the initial CutSet has very long cuts with a plenty of supervisions.
                # We cut these into smaller chunks centered around each supervision, possibly adding acoustic
                # context.
                print(f"About to split {partition} raw cuts into smaller chunks.")
                cut_set = cut_set.trim_to_supervisions(
                    keep_overlapping=False,
                    min_duration=None
                    if args.context_window <= 0.0
                    else args.context_window,
                    context_direction=args.context_direction,
                )
                if partition in ["L", "XL"]:
                    # Before storing manifests in the arrow format, we want to pre-shuffle them,
                    # as the sampler won't be able to do it later in an efficient manner.
                    cut_set = cut_set.shuffle()

                if args.precomputed_features:
                    # Extract the features after cutting large recordings into smaller cuts.
                    # Note: we support very efficient "chunked" feature reads with the argument
                    #       `storage_type=ChunkedLilcomHdf5Writer`, but we don't support efficient
                    #       data augmentation and feature computation for long recordings yet.
                    #       Therefore, we sacrifice some storage for the ability to precompute
                    #       features on shorter chunks, without memory blow-ups.
                    cut_set = cut_set.compute_and_store_features(
                        extractor=extractor,
                        storage_path=f"{output_dir}/feats_gigaspeech_{partition}",
                        # when an executor is specified, make more partitions
                        num_jobs=args.num_jobs if ex is None else 80,
                        executor=ex,
                    )

                cut_set.to_file(cuts_path)

                # Remove cut_set so the next iteration can correctly infer whether it needs to
                # load the raw cuts from disk or not.
                del cut_set

        # Now onto Musan
        if not musan_cuts_path.is_file():
            print("Extracting features for Musan")
            # create chunks of Musan with duration 5 - 10 seconds
            musan_cuts = (
                CutSet.from_manifests(
                    recordings=combine(
                        part["recordings"] for part in musan_manifests.values()
                    )
                )
                .cut_into_windows(10.0)
                .filter(lambda c: c.duration > 5)
                .compute_and_store_features(
                    extractor=extractor,
                    storage_path=f"{output_dir}/feats_musan",
                    num_jobs=args.num_jobs if ex is None else 80,
                    executor=ex,
                    storage_type=LilcomHdf5Writer,
                )
            )
            musan_cuts.to_file(musan_cuts_path)


if __name__ == "__main__":
    main()
