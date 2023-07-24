# Copyright      2023  Xiaomi Corp.        (authors: Zengrui Jin)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


import glob
import logging
import re
from pathlib import Path
from typing import List, Dict

import lhotse
from lhotse import CutSet, load_manifest_lazy


class MultiDataset:
    def __init__(self, fbank_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:
            - aidatatang_cuts_train.jsonl.gz
            - aishell_cuts_train.jsonl.gz
            - aishell2_cuts_train.jsonl.gz
            - aishell4_cuts_train_L.jsonl.gz
            - aishell4_cuts_train_M.jsonl.gz
            - aishell4_cuts_train_S.jsonl.gz
            - alimeeting-far_cuts_train.jsonl.gz
            - magicdata_cuts_train.jsonl.gz
            - primewords_cuts_train.jsonl.gz
            - stcmds_cuts_train.jsonl.gz
            - thchs_30_cuts_train.jsonl.gz
            - kespeech/kespeech-asr_cuts_train_phase1.jsonl.gz
            - kespeech/kespeech-asr_cuts_train_phase2.jsonl.gz
            - wenetspeech/cuts_L.jsonl.gz
        """
        self.fbank_dir = Path(fbank_dir)

    def train_cuts(self) -> CutSet:
        logging.info("About to get multidataset train cuts")

        # THCHS-30
        logging.info("Loading THCHS-30 in lazy mode")
        thchs_30_cuts = load_manifest_lazy(
            self.fbank_dir / "thchs_30_cuts_train.jsonl.gz"
        )

        # AISHELL-1
        logging.info("Loading Aishell-1 in lazy mode")
        aishell_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_train.jsonl.gz"
        )

        # AISHELL-2
        logging.info("Loading Aishell-2 in lazy mode")
        aishell_2_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_train.jsonl.gz"
        )

        # AISHELL-4
        logging.info("Loading Aishell-4 in lazy mode")
        aishell_4_L_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell4_cuts_train_L.jsonl.gz"
        )
        aishell_4_M_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell4_cuts_train_M.jsonl.gz"
        )
        aishell_4_S_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell4_cuts_train_S.jsonl.gz"
        )

        # ST-CMDS
        logging.info("Loading ST-CMDS in lazy mode")
        stcmds_cuts = load_manifest_lazy(self.fbank_dir / "stcmds_cuts_train.jsonl.gz")

        # Primewords
        logging.info("Loading Primewords in lazy mode")
        primewords_cuts = load_manifest_lazy(
            self.fbank_dir / "primewords_cuts_train.jsonl.gz"
        )

        # MagicData
        logging.info("Loading MagicData in lazy mode")
        magicdata_cuts = load_manifest_lazy(
            self.fbank_dir / "magicdata_cuts_train.jsonl.gz"
        )

        # Aidatatang_200zh
        logging.info("Loading Aidatatang_200zh in lazy mode")
        aidatatang_200zh_cuts = load_manifest_lazy(
            self.fbank_dir / "aidatatang_cuts_train.jsonl.gz"
        )

        # Ali-Meeting
        logging.info("Loading Ali-Meeting in lazy mode")
        alimeeting_cuts = load_manifest_lazy(
            self.fbank_dir / "alimeeting-far_cuts_train.jsonl.gz"
        )

        # WeNetSpeech
        logging.info("Loading WeNetSpeech in lazy mode")
        wenetspeech_L_cuts = load_manifest_lazy(
            self.fbank_dir / "wenetspeech" / "cuts_L.jsonl.gz"
        )

        # KeSpeech
        logging.info("Loading KeSpeech in lazy mode")
        kespeech_1_cuts = load_manifest_lazy(
            self.fbank_dir / "kespeech" / "kespeech-asr_cuts_train_phase1.jsonl.gz"
        )
        kespeech_2_cuts = load_manifest_lazy(
            self.fbank_dir / "kespeech" / "kespeech-asr_cuts_train_phase2.jsonl.gz"
        )

        return CutSet.mux(
            thchs_30_cuts,
            aishell_cuts,
            aishell_2_cuts,
            aishell_4_L_cuts,
            aishell_4_M_cuts,
            aishell_4_S_cuts,
            stcmds_cuts,
            primewords_cuts,
            magicdata_cuts,
            aidatatang_200zh_cuts,
            alimeeting_cuts,
            wenetspeech_L_cuts,
            kespeech_1_cuts,
            kespeech_2_cuts,
            weights=[
                len(thchs_30_cuts),
                len(aishell_cuts),
                len(aishell_2_cuts),
                len(aishell_4_L_cuts),
                len(aishell_4_M_cuts),
                len(aishell_4_S_cuts),
                len(stcmds_cuts),
                len(primewords_cuts),
                len(magicdata_cuts),
                len(aidatatang_200zh_cuts),
                len(alimeeting_cuts),
                len(wenetspeech_L_cuts),
                len(kespeech_1_cuts),
                len(kespeech_2_cuts),
            ],
        )

    def dev_cuts(self) -> CutSet:
        logging.info("About to get multidataset dev cuts")

        # Aidatatang_200zh
        logging.info("Loading Aidatatang_200zh DEV set in lazy mode")
        aidatatang_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aidatatang_cuts_dev.jsonl.gz"
        )

        # AISHELL
        logging.info("Loading Aishell DEV set in lazy mode")
        aishell_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_dev.jsonl.gz"
        )

        # AISHELL-2
        logging.info("Loading Aishell-2 DEV set in lazy mode")
        aishell2_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_dev.jsonl.gz"
        )

        # Ali-Meeting
        logging.info("Loading Ali-Meeting DEV set in lazy mode")
        alimeeting_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "alimeeting-far_cuts_eval.jsonl.gz"
        )

        # MagicData
        logging.info("Loading MagicData DEV set in lazy mode")
        magicdata_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "magicdata_cuts_dev.jsonl.gz"
        )

        # KeSpeech
        logging.info("Loading KeSpeech DEV set in lazy mode")
        kespeech_dev_phase1_cuts = load_manifest_lazy(
            self.fbank_dir / "kespeech" / "kespeech-asr_cuts_dev_phase1.jsonl.gz"
        )
        kespeech_dev_phase2_cuts = load_manifest_lazy(
            self.fbank_dir / "kespeech" / "kespeech-asr_cuts_dev_phase2.jsonl.gz"
        )

        # WeNetSpeech
        logging.info("Loading WeNetSpeech DEV set in lazy mode")
        wenetspeech_dev_cuts = load_manifest_lazy(
            self.fbank_dir / "wenetspeech" / "cuts_DEV.jsonl.gz"
        )

        return wenetspeech_dev_cuts
        # return [
        #         aidatatang_dev_cuts,
        #         aishell_dev_cuts,
        #         aishell2_dev_cuts,
        #         alimeeting_dev_cuts,
        #         magicdata_dev_cuts,
        #         kespeech_dev_phase1_cuts,
        #         kespeech_dev_phase2_cuts,
        #         wenetspeech_dev_cuts,
        #     ]

    def test_cuts(self) -> Dict[str, CutSet]:
        logging.info("About to get multidataset test cuts")

        # Aidatatang_200zh
        logging.info("Loading Aidatatang_200zh TEST set in lazy mode")
        aidatatang_test_cuts = load_manifest_lazy(
            self.fbank_dir / "aidatatang_cuts_test.jsonl.gz"
        )

        # AISHELL
        logging.info("Loading Aishell TEST set in lazy mode")
        aishell_test_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell_cuts_test.jsonl.gz"
        )

        # AISHELL-2
        logging.info("Loading Aishell-2 TEST set in lazy mode")
        aishell2_test_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell2_cuts_test.jsonl.gz"
        )

        # AISHELL-4
        logging.info("Loading Aishell-4 TEST set in lazy mode")
        aishell4_test_cuts = load_manifest_lazy(
            self.fbank_dir / "aishell4_cuts_test.jsonl.gz"
        )

        # Ali-Meeting
        logging.info("Loading Ali-Meeting TEST set in lazy mode")
        alimeeting_test_cuts = load_manifest_lazy(
            self.fbank_dir / "alimeeting-far_cuts_test.jsonl.gz"
        )

        # MagicData
        logging.info("Loading MagicData TEST set in lazy mode")
        magicdata_test_cuts = load_manifest_lazy(
            self.fbank_dir / "magicdata_cuts_test.jsonl.gz"
        )

        # KeSpeech
        logging.info("Loading KeSpeech TEST set in lazy mode")
        kespeech_test_cuts = load_manifest_lazy(
            self.fbank_dir / "kespeech" / "kespeech-asr_cuts_test.jsonl.gz"
        )

        # WeNetSpeech
        logging.info("Loading WeNetSpeech TEST set in lazy mode")
        wenetspeech_test_meeting_cuts = load_manifest_lazy(
            self.fbank_dir / "wenetspeech" / "cuts_TEST_MEETING.jsonl.gz"
        )
        wenetspeech_test_net_cuts = load_manifest_lazy(
            self.fbank_dir / "wenetspeech" / "cuts_TEST_NET.jsonl.gz"
        )

        return {
            "aidatatang": aidatatang_test_cuts,
            # "alimeeting": alimeeting_test_cuts,
            "aishell": aishell_test_cuts,
            "aishell-2": aishell2_test_cuts,
            "aishell-4": aishell4_test_cuts,
            "magicdata": magicdata_test_cuts,
            "kespeech": kespeech_test_cuts,
            "wenetspeech-meeting": wenetspeech_test_meeting_cuts,
            "wenetspeech-net": wenetspeech_test_net_cuts,
        }
