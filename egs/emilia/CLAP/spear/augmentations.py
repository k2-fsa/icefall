import random
import time

from lhotse.cut import CutSet, MonoCut, Cut
from lhotse.cut.set import mix

def _mix_with_offset_deprecated_(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    snr: float = 10.0,
    drop_mixed_in_supervision: bool = True
):
    if drop_mixed_in_supervision:
        mixed_in_cut = mixed_in_cut.drop_supervisions()
    ref_duration = reference_cut.duration
    mixed_in_duration = mixed_in_cut.duration
    
    mix_duration = random.uniform(0.1, ref_duration / 2) # 0.1 for safety
    
    # randomly truncate the mixed_in_cut to mix_duration if longer
    if mixed_in_duration > mix_duration:
        diff = max(0.0, mixed_in_duration - mix_duration - 0.05)
        truncate_start = random.uniform(0, diff)
        mixed_in_cut = mixed_in_cut.truncate(offset=truncate_start, duration=mix_duration)
        
    actual_mix_duration = min(mixed_in_cut.duration, mix_duration)
    offset = random.uniform(0, ref_duration - actual_mix_duration - 0.05) # a tolerance of 0.05 for safety
    mixed_cut = mix(
        reference_cut=reference_cut,
        mixed_in_cut=mixed_in_cut,
        offset=offset,
        snr=snr,
        preserve_id="left",
    )
    
    return mixed_cut

def mix_with_offset(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    snr: float = 10.0,
    drop_mixed_in_supervision: bool = True,
    *,
    # 仅对“语音重叠”模式——如果是噪声注入，建议另行分支处理
    min_overlap_ratio: float = 0.20,   # 下限
    max_overlap_ratio: float = 0.50,   # 上限
    epsilon: float = 0.01,             # tolerance
):
    if drop_mixed_in_supervision and hasattr(mixed_in_cut, "drop_supervisions"):
        mixed_in_cut = mixed_in_cut.drop_supervisions()

    ref_duration = float(reference_cut.duration)
    if ref_duration <= (0.1 + epsilon):
        return reference_cut  # 极短段保护

    # 计算严格 < 50% 的上界
    max_allowed = max(0.0, max_overlap_ratio * ref_duration - epsilon)
    min_allowed = max(0.1, min_overlap_ratio * ref_duration)  # 0.1s 安全下限与原逻辑一致
    if max_allowed < min_allowed:  # 容错：极短段或参数不当
        min_allowed = max(0.05, min_allowed * 0.5)
        max_allowed = max(min_allowed + epsilon, max_allowed)

    mix_duration = random.uniform(min_allowed, max_allowed)

    # 截断被混入段以满足目标 mix_duration
    mixed_in_duration = float(mixed_in_cut.duration)
    if mixed_in_duration > mix_duration:
        # 在可行范围内随机起点后截断
        slack = max(0.0, mixed_in_duration - mix_duration - epsilon)
        truncate_start = random.uniform(0.0, slack) if slack > 0 else 0.0
        mixed_in_cut = mixed_in_cut.truncate(offset=truncate_start, duration=mix_duration)

    actual_mix_duration = min(float(mixed_in_cut.duration), mix_duration)

    # 将被混入段完全放入参考段内部，保证真实 overlap = actual_mix_duration
    hi = max(0.0, ref_duration - actual_mix_duration - epsilon)
    offset = random.uniform(0.0, hi) if hi > 0 else 0.0

    mixed_cut = mix(
        reference_cut=reference_cut,
        mixed_in_cut=mixed_in_cut,
        offset=offset,
        snr=snr,
        preserve_id="left",
    )
    return mixed_cut


class BatchMixing:
    def __init__(
        self,
        min_snr: float = -5,
        max_snr: float = 5,
        min_noise_snr: float = -5,
        p: float = 0.2,
        p_noise: float = 0.1,
        noise_cuts: CutSet = None,
        drop_mixed_in_supervision: bool = True,
        seed: int = 42,
        stateful: bool = True,
    ):
        """perform in-batch mixing with the cuts from the same batch

        Args:
            min_snr (float): minimum mix SNR for in-batch speech mixing
            max_snr (float): maximum mix SNR
            min_noise_snr (float): minimum mix SNR for noise mixing
            p_noise (float, optional): The probability of perform noise mixing instead of in-batch.
            p (float, optional): The probability of perform mixing to a cut. Defaults to 0.5.
            noise_cuts (CutSet, optional): An optional noise cut. If provided, sample from the noise cuts instead of from the batch itself
            drop_mixed_in_supervision (bool, optional): Remove the supervisions in the mixed_in_cut. Defaults to True.
        """
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p
        self.min_noise_snr = min_noise_snr
        self.p_noise = p_noise
        if p_noise > 0:
            assert noise_cuts is not None, "If p_noise > 0, noise_cuts must be provided"
        self.noise_cuts = noise_cuts
        self.drop_mixed_in_supervision = drop_mixed_in_supervision
        
        self.seed = seed
        self.stateful = stateful
        self.num_times_iterated = 0
    
    def __str__(self):
        return f"BatchMixing: p={self.p}, snr=({self.min_snr}, {self.max_snr}), p_n={self.p_noise}, min_noise_snr={self.min_noise_snr}, drop_supervision={self.drop_mixed_in_supervision}" 
        
    def __call__(self, reference_cuts: CutSet) -> CutSet:
        from lhotse.dataset.dataloading import resolve_seed

        if isinstance(self.seed, random.Random):
            rng = self.seed
        else:
            rng = random.Random(resolve_seed(self.seed) + self.num_times_iterated + int(time.time() * 1000) % 100000)

        if self.stateful:
            self.num_times_iterated += 1
        
        if self.noise_cuts.is_lazy:
            # If the noise input is lazy, we'll shuffle it approximately.
            # We set the shuffling buffer size to 2000 because that's the size of MUSAN,
            # so even if the user forgets to convert MUSAN to an eager manifest, they will
            # get roughly the same quality of noise randomness.
            # Note: we can't just call .to_eager() as the noise CutSet can technically be
            #       very large, or even hold data in-memory in case of webdataset/Lhotse Shar sources.
            def noise_gen():
                yield from self.noise_cuts.repeat().shuffle(rng=rng, buffer_size=2000)
        else:
            # Eager nose cuts are just fully reshuffled in a different order on each noise "epoch".
            def noise_gen():
                while True:
                    yield from self.noise_cuts.shuffle(rng=rng) 
        
        noise_cuts = iter(noise_gen())
        results = []
        for cut in reference_cuts:
            # perform augmentation
            if rng.random() < self.p:
                if self.p_noise > 0 and rng.random() < self.p_noise:
                    snr = rng.uniform(self.min_noise_snr, 20) # the max snr for noise mixing is 20dB
                    mixed_in_cut = next(noise_cuts)
                    mixed_cut = mix_with_offset(
                        cut,
                        mixed_in_cut,
                        snr=snr,
                        max_overlap_ratio=0.8,  # noise 可以覆盖更多
                    )
                    # mixed_in_cut = self.noise_cuts.sample(n_cuts=1) # this should be rather quick
                else: # same batch mixing
                    snr = rng.uniform(self.min_snr, self.max_snr)
                    mixed_in_cut = reference_cuts.sample(n_cuts=1) # this should be rather quick
                    while mixed_in_cut.id == cut.id:
                        mixed_in_cut = reference_cuts.sample(n_cuts=1)
                    mixed_cut = mix_with_offset(
                        cut,
                        mixed_in_cut,
                        snr=snr,
                        min_overlap_ratio=0.2,
                        max_overlap_ratio=0.5,
                    )
                results.append(mixed_cut)
            else:
                results.append(cut)
        return CutSet.from_cuts(results)
    
def _test_mix():
    from lhotse import load_manifest_lazy
    from lhotse import load_manifest
    manifest = "data/fbank/librispeech_cuts_dev-other.jsonl.gz"
    noise_cuts = "data/musan/noise_non_speech_musan_audioset.jsonl.gz"
    
    cuts = load_manifest_lazy(manifest).subset(first=200).drop_features()
    noise_cuts = load_manifest(noise_cuts).drop_features()
    
    from lhotse.cut import MixedCut
    transform = BatchMixing(
        min_snr=-5, 
        max_snr=5,
        p=0.2,
        min_noise_snr=5,
        p_noise=0.5,
        noise_cuts=noise_cuts,
        drop_mixed_in_supervision=True
    )
    # import pdb; pdb.set_trace()
    start = time.time()
    for i in range(1):
        mixed_cuts = transform(cuts)
        mix_durations = []
        for j, c in enumerate(mixed_cuts):
            if isinstance(c, MixedCut):
                mix_durations.append(c.tracks[1].cut.duration)
    end = time.time()
    print(f"Elasped: {end - start} seconds")
        # print(sum(mix_durations)/len(mix_durations))
            
    # print(mixed_cuts)
    
    # MixedCut(
    #     id='2067-143536-0050-22591_repeat0', 
    #     tracks=[
    #         MixTrack(cut=MonoCut(id='2067-143536-0050-22591_repeat0', start=0, duration=15.905, channel=0, supervisions=[SupervisionSegment(id='2067-143536-0050', recording_id='2067-143536-0050', start=0.0, duration=15.905, channel=0, text='AND MEN AND WOMEN MUTES WATCHING WITH HARD CURIOUS EYES THEN SEATED IN HER BARBARIC CHAIR ABOVE THEM ALL WITH MYSELF AT HER FEET WAS THE VEILED WHITE WOMAN WHOSE LOVELINESS AND AWESOME POWER SEEMED TO VISIBLY SHINE ABOUT HER LIKE A HALO', language='English', speaker='2067', gender=None, custom=None, alignment=None)], features=Features(type='kaldi-fbank', num_frames=1591, num_features=128, frame_shift=0.01, sampling_rate=16000, start=0, duration=15.905, storage_type='lilcom_chunky', storage_path='data/fbank/librispeech_feats_train-other-500/feats-1.lca', storage_key='272808444,75670,73700,73580,13443', recording_id='None', channels=0), recording=Recording(id='2067-143536-0050', sources=[AudioSource(type='file', channels=[0], source='/mnt/workspace/xiaoyu/workspace/icefall_prompt_multi_task/egs/librispeech/ASR/download/LibriSpeech/train-other-500/2067/143536/2067-143536-0050.flac')], sampling_rate=16000, num_samples=254480, duration=15.905, channel_ids=[0], transforms=None), custom={'codebook_indexes': TemporalArray(array=Array(storage_type='numpy_hdf5', storage_path='data_hdf5/vq_hubert_large_layer_21_normalize_1_cb_16/librispeech_cuts_train-all-shuf/librispeech_cuts_train-all-shuf-1.h5', storage_key='2067-143536-0050-22591', shape=[795, 16]), temporal_dim=0, frame_shift=0.02, start=0), 'shard_origin': PosixPath('data-shar/data-shar-hubert-large-layer-21-normalize-cb16-hdf5/librispeech/train-all-shuf/cuts.000083.jsonl.gz'), 'shar_epoch': 0, 'task_id': 1, 'dataloading_info': {'rank': 1, 'world_size': 8, 'worker_id': 1}}), type='MonoCut', offset=0.0, snr=None), 
    #         MixTrack(cut=MonoCut(id='4ed48ac9-a0df-e8e9-de68-e8540e51ae78', start=10.9971875, duration=0.58, channel=0, supervisions=[], features=Features(type='kaldi-fbank', num_frames=1588, num_features=128, frame_shift=0.01, sampling_rate=16000, start=0, duration=15.875, storage_type='lilcom_chunky', storage_path='data/fbank/librispeech_feats_train-other-500/feats-8.lca', storage_key='96050666,74833,75029,74339,13685', recording_id='None', channels=0), recording=Recording(id='8346-244446-0072', sources=[AudioSource(type='file', channels=[0], source='/mnt/workspace/xiaoyu/workspace/icefall_prompt_multi_task/egs/librispeech/ASR/download/LibriSpeech/train-other-500/8346/244446/8346-244446-0072.flac')], sampling_rate=16000, num_samples=254000, duration=15.875, channel_ids=[0], transforms=None), custom={'codebook_indexes': TemporalArray(array=Array(storage_type='numpy_hdf5', storage_path='data_hdf5/vq_hubert_large_layer_21_normalize_1_cb_16/librispeech_cuts_train-all-shuf/librispeech_cuts_train-all-shuf-3.h5', storage_key='8346-244446-0072-7733', shape=[793, 16]), temporal_dim=0, frame_shift=0.02, start=0), 'shard_origin': PosixPath('data-shar/data-shar-hubert-large-layer-21-normalize-cb16-hdf5/librispeech/train-all-shuf/cuts.000128.jsonl.gz'), 'shar_epoch': 0, 'task_id': 1, 'dataloading_info': {'rank': 1, 'world_size': 8, 'worker_id': 1}}), type='MonoCut', offset=7.163035072927992, snr=2.8853809253275147)], transforms=None)
    
if __name__=="__main__":
    _test_mix()