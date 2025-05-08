# https://huggingface.co/datasets/VocalNet/UltraChat-vocalnet/blob/main/UltraChat.json
# https://huggingface.co/datasets/VocalNet/VoiceAssistant-430K-vocalnet/blob/main/VoiceAssistant-430K.json
import json
import os

import numpy as np
from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.supervision import SupervisionSegment


class LazyCustomDatasetIterator:
    """
    Thin wrapper on top of HF datasets objects that allows to interact with them through a Lhotse CutSet.
    It can be initialized with an existing HF dataset, or args/kwargs passed on to ``datasets.load_dataset()``.
    Use ``audio_key``, ``text_key``, ``lang_key`` and ``gender_key`` options to indicate which keys in dict examples
    returned from HF Dataset should be looked up for audio, transcript, language, and gender respectively.
    The remaining keys in HF dataset examples will be stored inside ``cut.custom`` dictionary.
    Example with existing HF dataset::
        >>> import datasets
        ... dataset = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
        ... dataset = dataset.map(some_transform)
        ... cuts_it = LazyHFDatasetIterator(dataset)
        ... for cut in cuts_it:
        ...     pass
    Example providing HF dataset init args/kwargs::
        >>> import datasets
        ... cuts_it = LazyHFDatasetIterator("mozilla-foundation/common_voice_11_0", "hi", split="test")
        ... for cut in cuts_it:
        ...     pass
    """

    def __init__(self, json_file_path: str, shard_id: int = 0, num_shards: int = 100):
        self.json_file_path = json_file_path
        self.shard_id = shard_id
        self.num_shards = num_shards

    def __iter__(self):

        with open(self.json_file_path, "r", encoding="utf-8") as f:
            list_data_dict = json.load(f)
        list_data_dict = list_data_dict[self.shard_id :: self.num_shards]
        for item in list_data_dict:
            custom_data = item.copy()
            json_file_parent_of_parent_dir = os.path.dirname(
                os.path.dirname(self.json_file_path)
            )
            units_path = os.path.join(
                json_file_parent_of_parent_dir, custom_data["units"]
            )
            speech_token_dict = np.load(units_path, allow_pickle=True).item()
            speech_token = speech_token_dict["speech_token"].squeeze(0).tolist()
            speech_token_len = speech_token_dict["speech_token_len"]

            assert len(speech_token) == speech_token_len
            custom_data["speech_token"] = speech_token
            audio_path = custom_data.pop("speech", None)
            audio_path = os.path.join(json_file_parent_of_parent_dir, audio_path)
            item_id = item.get("id")
            recording = Recording.from_file(path=audio_path, recording_id=item_id)

            conversations = item.get("conversations")
            assert isinstance(conversations, list) and len(conversations) == 2
            for conv in conversations:
                if isinstance(conv, dict) and conv.get("from") == "gpt":
                    gpt_text = conv.get("value")
                    break
            assert gpt_text is not None

            supervision = SupervisionSegment(
                id=item_id,
                recording_id=recording.id,
                start=0.0,  # Assuming the supervision covers the entire recording
                duration=recording.duration,
                text=gpt_text,
            )

            cut = recording.to_cut()
            # cut.id will be the same as recording.id

            cut.supervisions = [supervision]
            # custom_data contains the original item's fields, minus "speech".
            # So, "id", "conversations", "units", etc., are preserved here.
            custom_data.pop("conversations")
            custom_data.pop("units")
            cut.custom = custom_data

            yield cut


if __name__ == "__main__":
    json_file_path = (
        "/workspace/slam/VoiceAssistant-430K-vocalnet/VoiceAssistant-430K.json"
    )
    cut_set = CutSet(LazyCustomDatasetIterator(json_file_path=json_file_path))

    for cut in cut_set:
        print(cut)
        input()
