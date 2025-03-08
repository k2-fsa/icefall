def get_zipformer_base_config():
    return {
        "output_downsampling_factor": 1,
        "downsampling_factor": (1, 2, 4, 8, 4, 2),
        "encoder_dim": (192, 256, 384, 512, 384, 256),
        "num_encoder_layers": (2, 2, 3, 4, 3, 2),
        "encoder_unmasked_dim": (192, 192, 256, 256, 256, 192),
        "query_head_dim": 32,
        "pos_head_dim": 4,
        "value_head_dim": 12,
        "pos_dim": 48,
        "num_heads": (4, 4, 4, 8, 4, 4),
        "feedforward_dim": (512, 768, 1024, 1536, 1024, 768),
        "cnn_module_kernel": (31, 31, 15, 15, 15, 31),
        "dropout": 0.1,
        "warmup_batches": 4000.0,
    }
