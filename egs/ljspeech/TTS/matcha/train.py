#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Fangjun Kuang)


import torch


from icefall.utils import AttributeDict
from matcha.models.matcha_tts import MatchaTTS


def get_model(params):
    m = MatchaTTS(**params.model)
    return m


def main():
    n_feats = 80
    filter_channels_dp = 256
    encoder_params_p_dropout = 0.1
    params = AttributeDict(
        {
            "model": AttributeDict(
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
        }
    )
    m = get_model(params)
    print(m)

    num_param = sum([p.numel() for p in m.parameters()])
    print(f"Number of parameters: {num_param}")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
