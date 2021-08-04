#!/usr/bin/env python3
#
# Copyright      2021  Xiaomi Corp.       (authors: Fangjun Kuang)
#
# See ../../../LICENSE for clarification regarding multiple authors
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

# To run this single test, use
#
#  ctest --verbose -R nbest_test_py

import unittest

import k2
import torch

from icefall.nbest import Nbest


class TestNbest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device('cpu')]
        if torch.cuda.is_available() and k2.with_cuda:
            cls.devices.append(torch.device('cuda', 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device('cuda', 1))

    def test_nbest_constructor(self):
        fsa = k2.Fsa.from_str('''
            0 1 -1 0.1
            1
        ''')

        fsa_vec = k2.create_fsa_vec([fsa, fsa, fsa])
        shape = k2.RaggedShape('[[x x] [x]]')
        Nbest(fsa_vec, shape)

    def test_top_k(self):
        fsa0 = k2.Fsa.from_str('''
            0 1 -1 0
            1
        ''')
        fsas = [fsa0.clone() for i in range(10)]
        fsa_vec = k2.create_fsa_vec(fsas)
        fsa_vec.scores = torch.tensor([3, 0, 1, 5, 4, 2, 8, 1, 9, 6],
                                      dtype=torch.float)
        #    0 1   2 3 4   5 6 7 8 9
        # [ [3 0] [1 5 4] [2 8 1 9 6]
        shape = k2.RaggedShape('[ [x x] [x x x] [x x x x x] ]')
        nbest = Nbest(fsa_vec, shape)

        # top_k: k is 1
        nbest1 = nbest.top_k(1)
        expected_fsa = k2.create_fsa_vec([fsa_vec[0], fsa_vec[3], fsa_vec[8]])
        assert str(nbest1.fsa) == str(expected_fsa)

        expected_shape = k2.RaggedShape('[ [x] [x] [x] ]')
        assert nbest1.shape == expected_shape

        # top_k: k is 2
        nbest2 = nbest.top_k(2)
        expected_fsa = k2.create_fsa_vec([
            fsa_vec[0], fsa_vec[1], fsa_vec[3], fsa_vec[4], fsa_vec[8],
            fsa_vec[6]
        ])
        assert str(nbest2.fsa) == str(expected_fsa)

        expected_shape = k2.RaggedShape('[ [x x] [x x] [x x] ]')
        assert nbest2.shape == expected_shape

        # top_k: k is 3
        nbest3 = nbest.top_k(3)
        expected_fsa = k2.create_fsa_vec([
            fsa_vec[0], fsa_vec[1], fsa_vec[1], fsa_vec[3], fsa_vec[4],
            fsa_vec[2], fsa_vec[8], fsa_vec[6], fsa_vec[9]
        ])
        assert str(nbest3.fsa) == str(expected_fsa)

        expected_shape = k2.RaggedShape('[ [x x x] [x x x] [x x x] ]')
        assert nbest3.shape == expected_shape

        # top_k: k is 4
        nbest4 = nbest.top_k(4)
        expected_fsa = k2.create_fsa_vec([
            fsa_vec[0], fsa_vec[1], fsa_vec[1], fsa_vec[1], fsa_vec[3],
            fsa_vec[4], fsa_vec[2], fsa_vec[2], fsa_vec[8], fsa_vec[6],
            fsa_vec[9], fsa_vec[5]
        ])
        assert str(nbest4.fsa) == str(expected_fsa)

        expected_shape = k2.RaggedShape('[ [x x x x] [x x x x] [x x x x] ]')
        assert nbest4.shape == expected_shape


if __name__ == '__main__':
    unittest.main()
