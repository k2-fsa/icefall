import torch
from asr_datamodule import _SeedWorkers
from lhotse import Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.dataset.sampling.dynamic_bucketing import FixedBucketBatchSizeConstraint
from torch.utils.data import DataLoader


def _test(rank, world_size, args, logs):
    cuts = load_manifest_lazy(args["manifest_path"]).filter(
        lambda c: 1 <= c.duration <= 20.0
    )

    constraint = FixedBucketBatchSizeConstraint(
        max_seq_len_buckets=args["max_seq_len_buckets"],
        batch_sizes=args["fixed_batch_sizes"],
    )

    sampler = DynamicBucketingSampler(
        cuts,
        constraint=constraint,
        shuffle=True,
        drop_last=True,
        duration_bins=args["duration_bins"],
        buffer_size=args["buffer_size"],
        world_size=world_size,
        rank=rank,
        sync_buckets=True,
        concurrent=False,
    )

    dataset = K2SpeechRecognitionDataset(
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=128))),
        return_cuts=True,
    )
    seed = torch.randint(0, 100000, ()).item()
    worker_init_fn = _SeedWorkers(seed)

    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=16,
        worker_init_fn=worker_init_fn,
    )

    for i, batch in enumerate(dl):
        cuts_in_batch = batch["supervisions"]["cut"]
        bs = len(cuts_in_batch)
        c0 = cuts_in_batch[0]
        bucket = constraint.select_bucket(constraint.max_seq_len_buckets, example=c0)
        shape = batch["inputs"].shape
        print(
            f"[rank {rank}/{world_size}] Step {i}, batch size={bs}, bucket={bucket}, shape={shape}",
            flush=True,
        )

        logs[rank].append((i, bs, bucket))


if __name__ == "__main__":
    from multiprocessing import Manager

    import torch.multiprocessing as mp

    max_duration = 1000
    world_size = 8
    seed = 42
    num_buckets = 30
    buffer_size = num_buckets * 5000
    manifest_path = "data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz"

    cuts = load_manifest_lazy(manifest_path).filter(lambda c: 1 <= c.duration <= 20.0)

    dummy_sampler = DynamicBucketingSampler(
        cuts,
        max_duration=max_duration,
        num_buckets=num_buckets,
        shuffle=True,
        drop_last=True,
        buffer_size=buffer_size,
        world_size=world_size,
        rank=0,
        seed=seed,
        sync_buckets=True,
        concurrent=False,
    )
    duration_bins = dummy_sampler.duration_bins
    del dummy_sampler

    last_upper = 20.0  # + 1e-6
    max_seq_len_buckets = duration_bins + [last_upper]
    fixed_batch_sizes = [max(1, int(max_duration // ub)) for ub in max_seq_len_buckets]

    args = dict(
        manifest_path=manifest_path,
        duration_bins=duration_bins,
        max_seq_len_buckets=max_seq_len_buckets,
        fixed_batch_sizes=fixed_batch_sizes,
        buffer_size=buffer_size,
        seed=seed,
    )

    manager = Manager()
    logs = manager.dict({r: manager.list() for r in range(world_size)})

    mp.spawn(_test, args=(world_size, args, logs), nprocs=world_size, join=True)

    steps_list = [len(logs[r]) for r in range(world_size)]
    assert len(set(steps_list)) == 1, f"total steps mismatch across ranks: {steps_list}"
    total_steps = steps_list[0]

    for s in range(total_steps):
        batch_sizes = [logs[r][s][1] for r in range(world_size)]
        buckets = [logs[r][s][2] for r in range(world_size)]
        print(f"step {s}: batch size {batch_sizes}, bucket {buckets}")
        assert (
            len(set(batch_sizes)) == 1
        ), f"step {s}: batch size mismatch: {batch_sizes}"
        assert len(set(buckets)) == 1, f"step {s}: bucket mismatch: {buckets}"

    print(f"Done: verified {total_steps} steps")
