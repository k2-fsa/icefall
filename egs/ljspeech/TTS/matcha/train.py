#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Fangjun Kuang)


import torch


from icefall.utils import AttributeDict
from matcha.models.matcha_tts import MatchaTTS
from matcha.data.text_mel_datamodule import TextMelDataModule


def _get_data_params() -> AttributeDict:
    params = AttributeDict(
        {
            "name": "ljspeech",
            "train_filelist_path": "./filelists/ljs_audio_text_train_filelist.txt",
            "valid_filelist_path": "./filelists/ljs_audio_text_val_filelist.txt",
            "batch_size": 32,
            "num_workers": 3,
            "pin_memory": False,
            "cleaners": ["english_cleaners2"],
            "add_blank": True,
            "n_spks": 1,
            "n_fft": 1024,
            "n_feats": 80,
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "f_min": 0,
            "f_max": 8000,
            "seed": 1234,
            "load_durations": False,
            "data_statistics": AttributeDict(
                {
                    "mel_mean": -5.517028331756592,
                    "mel_std": 2.0643954277038574,
                }
            ),
        }
    )
    return params


def _get_model_params() -> AttributeDict:
    n_feats = 80
    filter_channels_dp = 256
    encoder_params_p_dropout = 0.1
    params = AttributeDict(
        {
            "n_vocab": 178,
            "n_spks": 1,  # for ljspeech.
            "spk_emb_dim": 64,
            "n_feats": n_feats,
            "out_size": None,  # or use 172
            "prior_loss": True,
            "use_precomputed_durations": False,
            "encoder": AttributeDict(
                {
                    "encoder_type": "RoPE Encoder",  # not used
                    "encoder_params": AttributeDict(
                        {
                            "n_feats": n_feats,
                            "n_channels": 192,
                            "filter_channels": 768,
                            "filter_channels_dp": filter_channels_dp,
                            "n_heads": 2,
                            "n_layers": 6,
                            "kernel_size": 3,
                            "p_dropout": encoder_params_p_dropout,
                            "spk_emb_dim": 64,
                            "n_spks": 1,
                            "prenet": True,
                        }
                    ),
                    "duration_predictor_params": AttributeDict(
                        {
                            "filter_channels_dp": filter_channels_dp,
                            "kernel_size": 3,
                            "p_dropout": encoder_params_p_dropout,
                        }
                    ),
                }
            ),
            "decoder": AttributeDict(
                {
                    "channels": [256, 256],
                    "dropout": 0.05,
                    "attention_head_dim": 64,
                    "n_blocks": 1,
                    "num_mid_blocks": 2,
                    "num_heads": 2,
                    "act_fn": "snakebeta",
                }
            ),
            "cfm": AttributeDict(
                {
                    "name": "CFM",
                    "solver": "euler",
                    "sigma_min": 1e-4,
                }
            ),
            "optimizer": AttributeDict(
                {
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                }
            ),
        }
    )

    return params


def get_params():
    params = AttributeDict(
        {
            "model": _get_model_params(),
            "data": _get_data_params(),
        }
    )
    return params


def get_model(params):
    m = MatchaTTS(**params.model)
    return m


def main():
    params = get_params()

    data_module = TextMelDataModule(hparams=params.data)
    if False:
        for b in data_module.train_dataloader():
            assert isinstance(b, dict)
            # b.keys()
            # ['x', 'x_lengths', 'y', 'y_lengths', 'spks', 'filepaths', 'x_texts', 'durations']
            # x: [batch_size, 289], torch.int64
            # x_lengths: [batch_size], torch.int64
            # y: [batch_size, n_feats, num_frames], torch.float32
            # y_lengths: [batch_size], torch.int64
            # spks: None
            # filepaths: list, (batch_size,)
            # x_texts: list, (batch_size,)
            # durations: None

    m = get_model(params)
    print(m)

    num_param = sum([p.numel() for p in m.parameters()])
    print(f"Number of parameters: {num_param}")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
