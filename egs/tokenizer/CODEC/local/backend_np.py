from io import BytesIO
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from lhotse.audio.backend import AudioBackend, FileObject
from lhotse.utils import Pathlike, Seconds


class WujiEEGBackend(AudioBackend):
    def read_audio(
        self,
        path_or_fd: Union[Pathlike, FileObject],
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        force_opus_sampling_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        np_arr = np.load(path_or_fd)
        sampling_rate = int(np_arr["fs"])
        return np_arr["eeg"][offset * sampling_rate : (offset + duration) * sampling_rate], sampling_rate 

    def is_applicable(self, path_or_fd: Union[Pathlike, FileObject]) -> bool:
        return True

    def supports_save(self) -> bool:
        return False

    def save_audio(
        self,
        dest: Union[str, Path, BytesIO],
        src: Union[torch.Tensor, np.ndarray],
        sampling_rate: int,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> None:
        raise NotImplementedError("Saving audio is not supported for the WujiEEGBackend.")

    def supports_info(self) -> bool:
        return True

    def info(
        self,
        path_or_fd: Union[Pathlike, FileObject],
    ):
        np_arr = np.load(path_or_fd)
        sampling_rate = int(np_arr["fs"])
        return NamedTuple(
            channels=1,
            frames=np_arr["eeg"].shape[0] // sampling_rate,
            samplerate=sampling_rate,
            duration=np_arr["eeg"].shape[0] / sampling_rate,
            video= None,
        )

