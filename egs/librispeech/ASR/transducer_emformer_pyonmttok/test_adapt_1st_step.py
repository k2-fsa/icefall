import torch
from icefall.utils import add_sos
import k2
from tokenizer import PyonmttokProcessor
from train_ubiqus_adaptation_step_2 import get_transducer_model
from icefall.utils import (
    AttributeDict,
)
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
from icefall.env import get_env_info
from icefall.checkpoint import load_checkpoint


def get_params() -> AttributeDict:

    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 100000,  # For the 100h subset, use 800
            "log_diagnostics": False,
            # parameters for Emformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "vgg_frontend": False,
            # parameters for decoder
            "embedding_dim": 2048,
            # parameters for Noam
            "warm_step": 160000,  # For the 100h subset, use 20000
            "env_info": get_env_info(),
            "exp_dir": "exp_ubiqus_context_15_lm_adaptation_bigger_lm",
            "context_size": 15,
            "attention_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            "left_context_length": 120,
            "segment_length": 16,
            "right_context_length": 4,
            "memory_size": 0,
            "start_batch": 0,
        }
    )

    return params


def load_checkpoint_if_available(
    params,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is positive, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      optimizer:
        The optimizer that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """

    filename = params.exp_dir + "/step2/checkpoint-3900000.pt"
    # filename = params.exp_dir +"/checkpoint-2168000.pt"

    saved_params = load_checkpoint(
        filename,
        model=model,
        optimizer=optimizer,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

        if "cur_batch_idx" in saved_params:
            params["cur_batch_idx"] = saved_params["cur_batch_idx"]

    return saved_params


def pred(model):
    from icefall.utils import add_sos
    import k2
    from tokenizer import PyonmttokProcessor
    device = "cpu"
    sp = PyonmttokProcessor()
    sp.load("../../ubiqus/ASR/data/lang_bpe_500/bpe.model")
    print(sp.piece_to_id("<blk>"))
    print(sp.piece_to_id("<unk>"))
    texts = ["Je fais une proposition"]
    # print(texts)
    y_list = sp.encode(texts, out_type=int)
    # print(y)
    y_list[0] = y_list[0]
    print(y_list)
    y = k2.RaggedTensor(y_list).to(device)
    blank_id = model.decoder.blank_id
    sos_y = add_sos(y, sos_id=blank_id)

    # sos_y_padded: [B, S + 1], start with SOS.
    sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
    y_padded = y.pad(mode="constant", padding_value=0)
    # decoder_out: [B, S + 1, C]

    with torch.no_grad():
        model.eval()
        decoder_out_fixed = model.decoder(sos_y_padded)
        print(decoder_out_fixed.shape)
        p_fixed = torch.exp(model.joiner.forward_lm(decoder_out_fixed))[
            :, :, :
        ].contiguous()

    print(p_fixed.shape)
    print(torch.max(p_fixed[0], dim=-1))
    print(p_fixed[0, :, 0])
    print(sp.decode(1 + torch.max(p_fixed[0], dim=-1).indices))


if __name__ == "__main__":
    sp = PyonmttokProcessor()
    sp.load("../../ubiqus/ASR/data/lang_bpe_500/bpe.model")
    params = get_params()

    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    model = get_transducer_model(params)

    pred(model)

    checkpoints = load_checkpoint_if_available(params=params, model=model)

    pred(model)
