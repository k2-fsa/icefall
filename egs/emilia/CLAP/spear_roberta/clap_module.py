import torch
import torch.distributed.nn
from torch import distributed as dist
from torch import nn as nn
from torch.nn import functional as F


def gather_features(
    audio_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_audio_features = torch.cat(
            torch.distributed.nn.all_gather(audio_features), dim=0
        )
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
        )
    else:
        gathered_audio_features = [
            torch.zeros_like(audio_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_audio_features, audio_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_audio_features[rank] = audio_features
            gathered_text_features[rank] = text_features

        all_audio_features = torch.cat(gathered_audio_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_audio_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size

    def forward(
        self,
        audio_features,
        text_features,
        logit_scale,
        multi_positive=False,
    ):
        device = audio_features.device

        if self.world_size > 1:
            all_audio_features, all_text_features = gather_features(
                audio_features=audio_features,
                text_features=text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
            )

            if self.local_loss:
                logits_per_audio = logit_scale * audio_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_audio_features.T
            else:
                logits_per_audio = (
                    logit_scale * all_audio_features @ all_text_features.T
                )
                logits_per_text = logits_per_audio.T
        else:
            logits_per_audio = logit_scale * audio_features @ text_features.T
            logits_per_text = logit_scale * text_features @ audio_features.T

        # calculated ground-truth
        if multi_positive:
            B_audio_local = audio_features.shape[0]
            B_text_local = text_features.shape[0]
            assert B_audio_local * 2 == B_text_local
            B = B_audio_local

            if not self.local_loss:
                num_audio_global = logits_per_audio.shape[0]
                idx_audio = torch.arange(num_audio_global, device=device)

                rank_audio = idx_audio // B
                local_audio = idx_audio % B

                pos1 = rank_audio * (2 * B) + local_audio
                pos2 = pos1 + B

                num_text_global = logits_per_text.shape[0]
                idx_text = torch.arange(num_text_global, device=device)

                rank_text = idx_text // (2 * B)
                labels_text = rank_text * B + idx_text % B
            else:
                idx_local_audio = torch.arange(B, device=device)
                pos1 = self.rank * (2 * B) + idx_local_audio
                pos2 = pos1 + B

                idx_local_text = torch.arange(2 * B, device=device)
                labels_text = self.rank * B + idx_local_text % B

            labels_audio = torch.zeros_like(logits_per_audio)
            labels_audio.scatter_(1, pos1.unsqueeze(1), 0.5)
            labels_audio.scatter_(1, pos2.unsqueeze(1), 0.5)

            total_loss = (
                F.cross_entropy(logits_per_audio, labels_audio)
                + F.cross_entropy(logits_per_text, labels_text)
            ) / 2

        else:
            num_logits = logits_per_audio.shape[0]
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank

            total_loss = (
                F.cross_entropy(logits_per_audio, labels)
                + F.cross_entropy(logits_per_text, labels)
            ) / 2

        return total_loss


def local_clip_loss(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    B = audio_features.shape[0]

    assert text_features.shape[0] == B
    assert text_features.shape[1] == 2

    logits = logit_scale * (audio_features.unsqueeze(1) * text_features).sum(dim=-1)

    # logsumexp(pos) = log(e^P1)
    log_sum_exp_pos = torch.logsumexp(logits[:, :1], dim=1)

    # logsumexp(all) = log(e^P1 + e^N1)
    log_sum_exp_all = torch.logsumexp(logits, dim=1)

    # Loss = - log ( sum(exp(pos)) / sum(exp(all)) )
    #      = - ( log(sum(exp(pos))) - log(sum(exp(all))) )
    #      = log_sum_exp_all - log_sum_exp_pos
    loss = log_sum_exp_all - log_sum_exp_pos

    return loss.mean()
