# Copyright      2021  Piotr Żelasko
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


import argparse
from typing import List, Union

from lhotse import CutSet
from torch.utils.data import DataLoader


class DataModule:
    """
    Contains dataset-related code. It is intended to read/construct Lhotse cuts,
    and create Dataset/Sampler/DataLoader out of them.

    There is a separate method to create each of train/valid/test DataLoader.
    In principle, there might be multiple DataLoaders for each of
    train/valid/test
    (e.g. when a corpus has multiple test sets).
    The API of this class allows to return lists of CutSets/DataLoaders.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        pass

    def train_cuts(self) -> Union[CutSet, List[CutSet]]:
        raise NotImplementedError()

    def valid_cuts(self) -> Union[CutSet, List[CutSet]]:
        raise NotImplementedError()

    def test_cuts(self) -> Union[CutSet, List[CutSet]]:
        raise NotImplementedError()

    def train_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()

    def valid_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()

    def test_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError()
