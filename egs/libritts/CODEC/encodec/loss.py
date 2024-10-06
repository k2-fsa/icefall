from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from lhotse.features.kaldi import Wav2LogFilterBank
from torchaudio.transforms import MelSpectrogram


class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "hinge",
    ):
        """Initialize GeneratorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(
        self,
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Calcualate generator adversarial loss.

        Args:
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs..

        Returns:
            Tensor: Generator adversarial loss value.

        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)

        return adv_loss / len(outputs)

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return F.relu(1 - x).mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "hinge",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)

        return real_loss / len(outputs), fake_loss / len(outputs)

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x.new_ones(x.size()) - x).mean()

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x.new_ones(x.size()) + x).mean()


class FeatureLoss(torch.nn.Module):
    """Feature loss module."""

    def __init__(
        self,
        average_by_layers: bool = True,
        average_by_discriminators: bool = True,
        include_final_outputs: bool = True,
    ):
        """Initialize FeatureMatchLoss module.

        Args:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            include_final_outputs (bool): Whether to include the final output of
                each discriminator for loss calculation.

        """
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(
        self,
        feats_hat: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        feats: Union[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate feature matching loss.

        Args:
            feats_hat (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from generator's outputs.
            feats (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from groundtruth..

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += (
                    (feat_hat_ - feat_).abs() / (feat_.abs().mean())
                ).mean()
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss / (len(feats) * len(feats[0]))


class MelSpectrogramReconstructionLoss(torch.nn.Module):
    """Mel Spec Reconstruction loss."""

    def __init__(
        self,
        sampling_rate: int = 22050,
        n_mels: int = 64,
        use_fft_mag: bool = True,
        return_mel: bool = False,
    ):
        super().__init__()
        self.wav_to_specs = []
        for i in range(5, 12):
            s = 2**i
            # self.wav_to_specs.append(
            #     Wav2LogFilterBank(
            #         sampling_rate=sampling_rate,
            #         frame_length=s,
            #         frame_shift=s // 4,
            #         use_fft_mag=use_fft_mag,
            #         num_filters=n_mels,
            #     )
            # )
            self.wav_to_specs.append(
                MelSpectrogram(
                    sample_rate=sampling_rate,
                    n_fft=s,
                    win_length=s,
                    hop_length=s // 4,
                    n_mels=n_mels,
                )
            )
        self.return_mel = return_mel

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Calculate Mel-spectrogram loss.

        Args:
            x_hat (Tensor): Generated waveform tensor (B, 1, T).
            x (Tensor): Groundtruth waveform tensor (B, 1, T).
            spec (Optional[Tensor]): Groundtruth linear amplitude spectrum tensor
                (B, T, n_fft // 2 + 1).  if provided, use it instead of groundtruth
                waveform.

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mel_loss = 0.0

        for i, wav_to_spec in enumerate(self.wav_to_specs):
            s = 2 ** (i + 5)
            wav_to_spec.to(x.device)

            mel_hat = wav_to_spec(x_hat.squeeze(1))
            mel = wav_to_spec(x.squeeze(1))

            mel_loss += F.l1_loss(
                mel_hat, mel, reduce=True, reduction="mean"
            ) + F.mse_loss(mel_hat, mel, reduce=True, reduction="mean")

        # mel_hat = self.wav_to_spec(x_hat.squeeze(1))
        # mel = self.wav_to_spec(x.squeeze(1))
        # mel_loss = F.l1_loss(mel_hat, mel) + F.mse_loss(mel_hat, mel)

        if self.return_mel:
            return mel_loss, (mel_hat, mel)

        return mel_loss


class WavReconstructionLoss(torch.nn.Module):
    """Wav Reconstruction loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate wav loss.

        Args:
            x_hat (Tensor): Generated waveform tensor (B, 1, T).
            x (Tensor): Groundtruth waveform tensor (B, 1, T).

        Returns:
            Tensor: Wav loss value.

        """
        wav_loss = F.l1_loss(x, x_hat, reduce=True, reduction="mean")

        return wav_loss


def adversarial_g_loss(y_disc_gen):
    """Hinge loss"""
    loss = 0.0
    for i in range(len(y_disc_gen)):
        stft_loss = F.relu(1 - y_disc_gen[i]).mean().squeeze()
        loss += stft_loss
    return loss / len(y_disc_gen)


def feature_loss(fmap_r, fmap_gen):
    loss = 0.0
    for i in range(len(fmap_r)):
        for j in range(len(fmap_r[i])):
            stft_loss = (
                (fmap_r[i][j] - fmap_gen[i][j]).abs() / (fmap_r[i][j].abs().mean())
            ).mean()
            loss += stft_loss
    return loss / (len(fmap_r) * len(fmap_r[0]))


def sim_loss(y_disc_r, y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_r)):
        loss += F.mse_loss(y_disc_r[i], y_disc_gen[i])
    return loss / len(y_disc_r)


def reconstruction_loss(x, x_hat, args, eps=1e-7):
    # NOTE (lsx): hard-coded now
    L = args.lambda_wav * F.mse_loss(x, x_hat)  # wav L1 loss
    # loss_sisnr = sisnr_loss(G_x, x) #
    # L += 0.01*loss_sisnr
    # 2^6=64 -> 2^10=1024
    # NOTE (lsx): add 2^11
    for i in range(6, 12):
        # for i in range(5, 12): # Encodec setting
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=args.sampling_rate,
            n_fft=max(s, 512),
            win_length=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": x_hat.device},
        ).to(x_hat.device)
        S_x = melspec(x)
        S_x_hat = melspec(x_hat)
        l1_loss = (S_x - S_x_hat).abs().mean()
        l2_loss = (
            ((torch.log(S_x.abs() + eps) - torch.log(S_x_hat.abs() + eps)) ** 2).mean(
                dim=-2
            )
            ** 0.5
        ).mean()

        alpha = (s / 2) ** 0.5
        L += l1_loss + alpha * l2_loss
    return L


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, args):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        print("last_layer cannot be none")
        assert 1 == 2
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 1.0, 1.0).detach()
    d_weight = d_weight * args.lambda_adv
    return d_weight


def loss_g(
    codebook_loss,
    speech,
    speech_hat,
    fmap,
    fmap_hat,
    y,
    y_hat,
    y_df,
    y_df_hat,
    y_ds,
    y_ds_hat,
    fmap_f,
    fmap_f_hat,
    fmap_s,
    fmap_s_hat,
    args=None,
):
    """
    args:
        codebook_loss: commit loss.
        speech: ground-truth wav.
        speech_hat: reconstructed wav.
        fmap: real stft-D feature map.
        fmap_hat: fake stft-D feature map.
        y: real stft-D logits.
        y_hat: fake stft-D logits.
        global_step: global training step.
        y_df: real MPD logits.
        y_df_hat: fake MPD logits.
        y_ds: real MSD logits.
        y_ds_hat: fake MSD logits.
        fmap_f: real MPD feature map.
        fmap_f_hat: fake MPD feature map.
        fmap_s: real MSD feature map.
        fmap_s_hat: fake MSD feature map.
    """
    rec_loss = reconstruction_loss(speech.contiguous(), speech_hat.contiguous(), args)
    adv_g_loss = adversarial_g_loss(y_hat)
    adv_mpd_loss = adversarial_g_loss(y_df_hat)
    adv_msd_loss = adversarial_g_loss(y_ds_hat)
    adv_loss = (
        adv_g_loss + adv_mpd_loss + adv_msd_loss
    ) / 3.0  # NOTE(lsx): need to divide by 3?
    feat_loss = feature_loss(
        fmap, fmap_hat
    )  # + sim_loss(y_disc_r, y_disc_gen) # NOTE(lsx): need logits?
    feat_loss_mpd = feature_loss(
        fmap_f, fmap_f_hat
    )  # + sim_loss(y_df_hat_r, y_df_hat_g)
    feat_loss_msd = feature_loss(
        fmap_s, fmap_s_hat
    )  # + sim_loss(y_ds_hat_r, y_ds_hat_g)
    feat_loss_tot = (feat_loss + feat_loss_mpd + feat_loss_msd) / 3.0
    d_weight = torch.tensor(1.0)

    # disc_factor = adopt_weight(
    #     args.lambda_adv, global_step, threshold=args.discriminator_iter_start
    # )
    disc_factor = 1
    if disc_factor == 0.0:
        fm_loss_wt = 0
    else:
        fm_loss_wt = args.lambda_feat

    loss = (
        rec_loss
        + d_weight * disc_factor * adv_loss
        + fm_loss_wt * feat_loss_tot
        + args.lambda_com * codebook_loss
    )
    return loss, rec_loss, adv_loss, feat_loss_tot, d_weight


if __name__ == "__main__":
    la = FeatureLoss(average_by_layers=False, average_by_discriminators=False)
    aa = [torch.rand(192, 192) for _ in range(3)]
    bb = [torch.rand(192, 192) for _ in range(3)]
    print(la(bb, aa))
    print(feature_loss(aa, bb))
