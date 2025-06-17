"""
Calculate pairwise Speaker Similarity betweeen two speech directories.
SV model wavlm_large_finetune.pth is downloaded from
    https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
SSL model wavlm_large.pt is downloaded from
    https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_large.pt
"""
import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-path", type=str, help="path of the evaluated speech directory"
    )
    parser.add_argument(
        "--test-list",
        type=str,
        help="path of the file list that contains the corresponding "
        "relationship between the prompt and evaluated speech. "
        "The first column is the wav name and the third column is the prompt speech",
    )
    parser.add_argument(
        "--sv-model-path",
        type=str,
        default="model/UniSpeech/wavlm_large_finetune.pth",
        help="path of the wavlm-based ECAPA-TDNN model",
    )
    parser.add_argument(
        "--ssl-model-path",
        type=str,
        default="model/s3prl/wavlm_large.pt",
        help="path of the wavlm SSL model",
    )
    return parser


class SpeakerSimilarity:
    def __init__(
        self,
        sv_model_path="model/UniSpeech/wavlm_large_finetune.pth",
        ssl_model_path="model/s3prl/wavlm_large.pt",
    ):
        """
        Initialize
        """
        self.sample_rate = 16000
        self.channels = 1
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logging.info("[Speaker Similarity] Using device: {}".format(self.device))
        self.model = ECAPA_TDNN_WAVLLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=16000,
            ssl_model_path=ssl_model_path,
        )
        state_dict = torch.load(
            sv_model_path, map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, wav_list, dtype="float32"):
        """
        Get embeddings
        """

        def _load_speech_task(fname, sample_rate):

            wav_data, sr = sf.read(fname, dtype=dtype)
            if sr != sample_rate:
                wav_data = librosa.resample(
                    wav_data, orig_sr=sr, target_sr=self.sample_rate
                )
            wav_data = torch.from_numpy(wav_data)

            return wav_data

        embd_lst = []
        for file_path in tqdm(wav_list):
            speech = _load_speech_task(file_path, self.sample_rate)
            speech = speech.to(self.device)
            with torch.no_grad():
                embd = self.model([speech])
            embd_lst.append(embd)

        return embd_lst

    def score(
        self,
        eval_path,
        test_list,
        dtype="float32",
    ):
        """
        Computes the Speaker Similarity (SIM-o) between two directories of speech files.

        Parameters:
        - eval_path (str): Path to the directory containing evaluation speech files.
        - test_list (str): Path to the file containing the corresponding relationship
             between prompt and evaluated speech.
        - dtype (str, optional): Data type for loading speech. Default is "float32".

        Returns:
        - float: The Speaker Similarity (SIM-o) score between the two directories
            of speech files.
        """
        prompt_wavs = []
        eval_wavs = []
        with open(test_list, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
                prompt_wavs.append(prompt_wav)
                eval_wavs.append(os.path.join(eval_path, wav_name + ".wav"))
        embds_prompt = self.get_embeddings(prompt_wavs, dtype=dtype)

        embds_eval = self.get_embeddings(eval_wavs, dtype=dtype)

        # Check if embeddings are empty
        if len(embds_prompt) == 0:
            logging.info("[Speaker Similarity] real set dir is empty, exiting...")
            return -1
        if len(embds_eval) == 0:
            logging.info("[Speaker Similarity] eval set dir is empty, exiting...")
            return -1

        scores = []
        for real_embd, eval_embd in zip(embds_prompt, embds_eval):
            scores.append(
                torch.nn.functional.cosine_similarity(real_embd, eval_embd, dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

        return np.mean(scores)


# part of the code is borrowed from https://github.com/lawlict/ECAPA-TDNN

""" Res2Conv1d + BatchNorm1d + ReLU
"""


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


""" Conv1d + BatchNorm1d + ReLU
"""


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


""" The SE connection of 1D case.
"""


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


""" SE-Res2Block of the ECAPA-TDNN architecture.
"""


# def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
#     return nn.Sequential(
#         Conv1dReluBn(channels, 512, kernel_size=1, stride=1, padding=0),
#         Res2Conv1dReluBn(512, kernel_size, stride, padding, dilation, scale=scale),
#         Conv1dReluBn(512, channels, kernel_size=1, stride=1, padding=0),
#         SE_Connect(channels)
#     )


class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels, kernel_size, stride, padding, dilation, scale=scale
        )
        self.Conv1dReluBn2 = Conv1dReluBn(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


""" Attentive weighted mean and standard deviation pooling.
"""


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear,
        #  then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, attention_channels, kernel_size=1
            )  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, attention_channels, kernel_size=1
            )  # equals W and b in the paper
        self.linear2 = nn.Conv1d(
            attention_channels, in_dim, kernel_size=1
        )  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN_WAVLLM(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        sr=16000,
        ssl_model_path=None,
    ):
        super().__init__()
        self.sr = sr

        if ssl_model_path is None:
            self.feature_extract = torch.hub.load("s3prl/s3prl", "wavlm_large")
        else:
            self.feature_extract = torch.hub.load(
                os.path.dirname(ssl_model_path),
                "wavlm_local",
                source="local",
                ckpt=ssl_model_path,
            )

        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[23].self_attn, "fp32_attention"
        ):
            self.feature_extract.model.encoder.layers[
                23
            ].self_attn.fp32_attention = False
        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[11].self_attn, "fp32_attention"
        ):
            self.feature_extract.model.encoder.layers[
                11
            ].self_attn.fp32_attention = False

        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )

        # self.conv = nn.Conv1d(self.channels[-1], self.channels[-1], kernel_size=1)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1],
            attention_channels=128,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features["hidden_states"]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        with torch.no_grad():
            x = self.feature_extract([sample for sample in x])

        x = x["hidden_states"]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        else:
            x = x.unsqueeze(0)
        norm_weights = (
            F.softmax(self.feature_weight, dim=-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    SIM = SpeakerSimilarity(
        sv_model_path=args.sv_model_path, ssl_model_path=args.ssl_model_path
    )
    score = SIM.score(args.eval_path, args.test_list)
    logging.info(f"SIM-o score: {score:.3f}")
