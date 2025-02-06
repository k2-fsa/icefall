from typing import Callable, Dict, List, Sequence, Union

import torch
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'features': (B x NumFrames x NumFeatures) float tensor
            'audio_lens': (B, ) int tensor
            'features_lens': (B, ) int tensor
            'text': List[str] of len B  # when return_text=True
            'tokens': List[List[str]]  # when return_tokens=True
            'speakers': List[str] of len B  # when return_spk_ids=True
            'cut': List of Cuts  # when return_cuts=True
        }
    """

    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        return_text: bool = True,
        return_tokens: bool = False,
        return_spk_ids: bool = False,
        return_cuts: bool = False,
    ) -> None:
        super().__init__()

        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        self.return_text = return_text
        self.return_tokens = return_tokens
        self.return_spk_ids = return_spk_ids
        self.return_cuts = return_cuts

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        # audio, audio_lens = collate_audio(cuts)
        features, features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            features = transform(features)

        batch = {
            #     "audio": audio,
            "features": features,
            #    "audio_lens": audio_lens,
            "features_lens": features_lens,
        }

        if self.return_text:
            # use normalized text
            # text = [cut.supervisions[0].normalized_text for cut in cuts]
            text = [cut.supervisions[0].text for cut in cuts]
            batch["text"] = text

        if self.return_tokens and "speech_tokens" in cuts[0].supervisions[0].custom:
            # tokens = [cut.tokens for cut in cuts]
            # tokens = [cut.supervisions[0].custom["tokens"]["text"] for cut in cuts]
            tokens = [cut.supervisions[0].custom["speech_tokens"] for cut in cuts]
            # change str into list
            tokens = [list(map(int, token.split())) for token in tokens]
            batch["tokens"] = tokens

        if self.return_spk_ids:
            batch["speakers"] = [cut.supervisions[0].speaker for cut in cuts]

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]

        return batch


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
