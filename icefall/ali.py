# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def save_alignments(
    alignments: Dict[str, List[int]],
    subsampling_factor: int,
    filename: str,
) -> None:
    """Save alignments to a file.

    Args:
      alignments:
        A dict containing alignments. Keys of the dict are utterances and
        values are the corresponding framewise alignments after subsampling.
      subsampling_factor:
        The subsampling factor of the model.
      filename:
        Path to save the alignments.
    Returns:
      Return None.
    """
    ali_dict = {
        "subsampling_factor": subsampling_factor,
        "alignments": alignments,
    }
    torch.save(ali_dict, filename)


def load_alignments(filename: str) -> Tuple[int, Dict[str, List[int]]]:
    """Load alignments from a file.

    Args:
      filename:
        Path to the file containing alignment information.
        The file should be saved by :func:`save_alignments`.
    Returns:
      Return a tuple containing:
        - subsampling_factor: The subsampling_factor used to compute
          the alignments.
        - alignments: A dict containing utterances and their corresponding
          framewise alignment, after subsampling.
    """
    ali_dict = torch.load(filename)
    subsampling_factor = ali_dict["subsampling_factor"]
    alignments = ali_dict["alignments"]
    return subsampling_factor, alignments


def convert_alignments_to_tensor(
    alignments: Dict[str, List[int]], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Convert alignments from list of int to a 1-D torch.Tensor.

    Args:
      alignments:
        A dict containing alignments. Keys are utterance IDs and
        values are their corresponding frame-wise alignments.
      device:
        The device to move the alignments to.
    Returns:
      Return a dict using 1-D torch.Tensor to store the alignments.
      The dtype of the tensor are `torch.int64`. We choose `torch.int64`
      because `torch.nn.functional.one_hot` requires that.
    """
    ans = {}
    for utt_id, ali in alignments.items():
        ali = torch.tensor(ali, dtype=torch.int64, device=device)
        ans[utt_id] = ali
    return ans


def lookup_alignments(
    cut_ids: List[str],
    alignments: Dict[str, torch.Tensor],
    num_classes: int,
    log_score: float = -10,
) -> torch.Tensor:
    """Return a mask constructed from alignments by a list of cut IDs.

    The returned mask is a 3-D tensor of shape (N, T, C). For each frame,
    i.e., each row, of the returned mask, positions not corresponding to
    the alignments are filled with `log_score`, while the position
    specified by the alignment is filled with 0. For instance, if the alignments
    of two utterances are:

        [ [1, 3, 2], [1, 0, 4, 2] ]
    num_classes is 5 and log_score is -10,  then the returned mask is

        [
          [[-10, 0, -10, -10, -10],
           [-10, -10, -10, 0, -10],
           [-10, -10, 0, -10, -10],
           [0, -10, -10, -10, -10]],
          [[-10, 0, -10, -10, -10],
           [0, -10, -10, -10, -10],
           [-10, -10, -10, -10, 0],
           [-10, -10, 0, -10, -10]]
        ]
    Note: We pad the alignment of the first utterance with 0.

    Args:
      cut_ids:
        A list of utterance IDs.
      alignments:
        A dict containing alignments. The keys are utterance IDs and the values
        are framewise alignments.
      num_classes:
        The max token ID + 1 that appears in the alignments.
      log_score:
        Positions in the returned tensor not corresponding to the alignments
        are filled with this value.
    Returns:
      Return a 3-D torch.float32 tensor of shape (N, T, C).
    """
    # We assume all utterances have their alignments.
    ali = [alignments[cut_id] for cut_id in cut_ids]
    padded_ali = pad_sequence(ali, batch_first=True, padding_value=0)
    padded_one_hot = torch.nn.functional.one_hot(
        padded_ali,
        num_classes=num_classes,
    )
    mask = (1 - padded_one_hot) * float(log_score)
    return mask
