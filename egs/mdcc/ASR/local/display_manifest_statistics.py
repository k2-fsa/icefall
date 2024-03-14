#!/usr/bin/env python3
# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengrui Jin,)
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

"""
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in transducer/train.py
for usage.
"""


from lhotse import load_manifest_lazy


def main():
    path = "./data/fbank/mdcc_cuts_train.jsonl.gz"
    path = "./data/fbank/mdcc_cuts_valid.jsonl.gz"
    path = "./data/fbank/mdcc_cuts_test.jsonl.gz"

    cuts = load_manifest_lazy(path)
    cuts.describe(full=True)


if __name__ == "__main__":
    main()

"""
data/fbank/mdcc_cuts_train.jsonl.gz (with speed perturbation)
_________________________________________ 
_ Cuts count:               _ 195360
_________________________________________            
_ Total duration (hh:mm:ss) _ 173:44:59
_________________________________________               
_ mean                      _ 3.2
_________________________________________
_ std                       _ 2.1
_________________________________________               
_ min                       _ 0.2
_________________________________________
_ 25%                       _ 1.8        
_________________________________________
_ 50%                       _ 2.7
_________________________________________
_ 75%                       _ 4.0
_________________________________________
_ 99%                       _ 11.0      _
_________________________________________
_ 99.5%                     _ 12.4      _
_________________________________________
_ 99.9%                     _ 14.8      _
_________________________________________
_ max                       _ 16.7      _
_________________________________________
_ Recordings available:     _ 195360    _
_________________________________________
_ Features available:       _ 195360    _
_________________________________________
_ Supervisions available:   _ 195360    _
_________________________________________

data/fbank/mdcc_cuts_valid.jsonl.gz 
________________________________________ 
_ Cuts count:               _ 5663     _ 
________________________________________ 
_ Total duration (hh:mm:ss) _ 05:03:12 _ 
________________________________________ 
_ mean                      _ 3.2      _ 
________________________________________ 
_ std                       _ 2.0      _ 
________________________________________ 
_ min                       _ 0.3      _ 
________________________________________ 
_ 25%                       _ 1.8      _ 
________________________________________ 
_ 50%                       _ 2.7      _ 
________________________________________ 
_ 75%                       _ 4.0      _ 
________________________________________ 
_ 99%                       _ 10.9     _ 
________________________________________
_ 99.5%                     _ 12.3     _
________________________________________
_ 99.9%                     _ 14.4     _
________________________________________
_ max                       _ 14.8     _
________________________________________
_ Recordings available:     _ 5663     _
________________________________________
_ Features available:       _ 5663     _
________________________________________
_ Supervisions available:   _ 5663     _
________________________________________

data/fbank/mdcc_cuts_test.jsonl.gz
________________________________________ 
_ Cuts count:               _ 12492    _ 
________________________________________ 
_ Total duration (hh:mm:ss) _ 11:00:31 _ 
________________________________________ 
_ mean                      _ 3.2      _ 
________________________________________ 
_ std                       _ 2.0      _ 
________________________________________ 
_ min                       _ 0.2      _ 
________________________________________ 
_ 25%                       _ 1.8      _ 
________________________________________ 
_ 50%                       _ 2.7      _ 
________________________________________ 
_ 75%                       _ 4.0      _ 
________________________________________ 
_ 99%                       _ 10.5     _ 
________________________________________ 
_ 99.5%                     _ 12.1     _ 
________________________________________
_ 99.9%                     _ 14.0     _
________________________________________
_ max                       _ 14.8     _
________________________________________
_ Recordings available:     _ 12492    _
________________________________________
_ Features available:       _ 12492    _
________________________________________
_ Supervisions available:   _ 12492    _
________________________________________

"""
