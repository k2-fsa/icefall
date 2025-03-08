# 預訓練參數設置
train_params = {
    # 模型參數
    "label_rate": 50,
    "sample_rate": 16000,
    "extractor_mode": "default",
    "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
    "conv_bias": False,
    "feature_grad_mult": 1.0,
    
    # 掩碼參數
    "mask_length": 10,
    "mask_prob": 0.65,
    "mask_selection": "static",
    "mask_other": 0,
    "no_mask_overlap": False,
    "mask_min_space": 1,
    
    # 通道掩碼參數
    "mask_channel_length": 10,
    "mask_channel_prob": 0.0,
    "mask_channel_selection": "static",
    "mask_channel_other": 0,
    "no_mask_channel_overlap": False,
    "mask_channel_min_space": 1,
    
    # 損失計算參數
    "skip_masked": False,
    "skip_nomask": False,
    "pred_masked_weight": 1,
    "pred_nomask_weight": 0,
    "loss_weights": [10],
    "checkpoint_activations": False,
    
    # 其他參數
    "dropout_input": 0.0,
    "dropout_features": 0.0,
    "num_classes": [504],
    "untie_final_proj": False,
    "required_seq_len_multiple": 2,
    "logit_temp": 0.1,
}
