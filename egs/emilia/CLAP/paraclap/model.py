import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model


class Projection(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out, bias=False)
        self.linear2 = torch.nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_out)
        self.drop = torch.nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class SpeechEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.base = Wav2Vec2Model.from_pretrained(self.model_name)
        self.hidden_size = self.base.config.hidden_size

    def forward(self, x):
        x = self.base(x)["last_hidden_state"]
        x = x.mean(1)
        return x


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :].detach()  # get CLS token output
        return out


class CLAP(torch.nn.Module):
    def __init__(self, speech_name: str, text_name: str, embedding_dim: int = 1024):
        super().__init__()

        self.audio_branch = SpeechEncoder(model_name=speech_name)

        self.text_branch = TextEncoder(model_name=text_name)
        self.audio_projection = Projection(self.audio_branch.hidden_size, embedding_dim)
        self.text_projection = Projection(
            self.text_branch.base.config.hidden_size, embedding_dim
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, text):
        speech_emb = self.audio_branch(audio)
        text_emb = self.text_branch(text)

        speech_emb = self.audio_projection(speech_emb)
        text_emb = self.text_projection(text_emb)

        return text_emb, speech_emb, self.logit_scale.exp()

    def forward_audio_branch(self, audio):
        speech_emb = self.audio_branch(audio)
        speech_emb = self.audio_projection(speech_emb)

        return speech_emb

    def forward_text_branch(self, text):
        text_emb = self.text_branch(text)
        text_emb = self.text_projection(text_emb)

        return text_emb
