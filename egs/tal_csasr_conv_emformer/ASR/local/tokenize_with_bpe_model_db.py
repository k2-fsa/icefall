#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright    2021 Mobvoi Inc.          (authors: Binbin Zhang)
# Copyright    2022  Xiaomi Corp.        (authors: Mingshuang Luo)
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
This script takes as input text (it includes Chinese and English):
    - text
and generates the text_with_bpe.
    - text_with_bpe
"""


import argparse
import logging

import sentencepiece as spm
from tqdm import tqdm

#from icefall.utils import tokenize_by_bpe_model

import re
from bpemb import BPEmb
from bpemb.util import sentencepiece_load, load_word2vec_file


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare text_with_bpe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/lang_char/text",
        type=str,
        help="the text includes Chinese and English words",
    )
    parser.add_argument(
        "--output",
        default="data/lang_char/text_with_bpe",
        type=str,
        help="the text_with_bpe tokenized by bpe model",
    )
    parser.add_argument(
        "--bpe-model",
        default="data/lang_char/bpe.model",
        type=str,
        help="the bpe model for processing the English parts",
    )

    return parser

def splitOOV(word,dic):
  #假设oov最长的元素为7
  Len=len(word)
  temp_syllable=[]
  temp_syllable_id=[]
  i=0
  word=word.strip() 
  while i < len(word):
     if word[i:i + 7] in dic:
        temp_syllable.append(word[i:i + 7])
        temp_syllable_id.append(dic[word[i:i + 7]])
        i += 7
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 6] in dic:
        temp_syllable.append(word[i:i + 6])
        temp_syllable_id.append(dic[word[i:i + 6]])
        i += 6
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 5] in dic:
        temp_syllable.append(word[i:i + 5])
        temp_syllable_id.append(dic[word[i:i + 5]])
        i += 5
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 4] in dic:
        temp_syllable.append(word[i:i + 4])
        temp_syllable_id.append(dic[word[i:i + 4]])
        i += 4
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 3] in dic:
        temp_syllable.append(word[i:i + 3])
        temp_syllable_id.append(dic[word[i:i + 3]])
        i += 3
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 2] in dic:
        temp_syllable.append(word[i:i + 2])
        temp_syllable_id.append(dic[word[i:i + 2]])
        i += 2
        continue  # 跳出本次循环，继续下一个
     elif word[i:i + 1] in dic:
        temp_syllable.append(word[i:i + 1])
        temp_syllable_id.append(dic[word[i:i + 1]])
        i += 1
        continue  # 跳出本次循环，继续下一个
     else:
        #print('still OOV',word[i],word)
        i += 1
        continue
  return temp_syllable,temp_syllable_id

def getsyllabelId(sylableId,cmufile):
  sylabbleId={}
  with open(sylableId,'r',encoding='utf-8') as fr:
     for line in fr:
        splitline=line.strip().split(" ")
        key = splitline[0]
        if key not in sylabbleId:
          sylabbleId[key]=splitline[1]
  newCmu={} 
  #with open('cmuId','w',encoding='utf-8') as fw:
  with open(cmufile,'r',encoding='utf-8') as fr:
       for line in fr:
          ids=[]
          splitline=re.split("\s+",line.strip().upper(),1) #line.strip().split("\t")
          key=splitline[0]
          syllable=splitline[1].split(" ")
          for ele in syllable:
             if ele in sylabbleId:
                ids.append(ele)
             else:
               if ele !=' ': 
                 spEle,spEleId=splitOOV(ele,sylabbleId)
                 ids.append(' '.join(spEle))
          if key not in newCmu:
            newCmu[key]=' '.join(ids)
            #fw.write(key+"\t"+splitline[1]+'\t'+' '.join(ids)+'\n')
   
  return sylabbleId,newCmu

def tokenize_by_bpe_model(content):
      sylableId,cmufile="data/lang_char/tokens.txt","BpeDict/dict/lexicon.out"
      ChToken,EnToken=getsyllabelId(sylableId,cmufile)
      tokens=[]
      pattern = re.compile(r"([\u4e00-\u9fff])")
      chars = pattern.split(content)
      mix_chars = [w for w in chars if len(w.strip()) > 0]
      for ch_or_w in mix_chars:
        if pattern.fullmatch(ch_or_w) is not None:
           tokens.append(ch_or_w)
        else:
           for ch  in ch_or_w.upper().split(" "):
              if ch in EnToken:
                  tokens.append(EnToken[ch])
              else:
                  pass
      return ' '.join(tokens)


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    #bpe_model = args.bpe_model

    
    

    f = open(input_file, "r", encoding="utf-8")
    lines = f.readlines()

    logging.info("Starting reading the text")
    f_out = open(output_file, "w", encoding="utf-8")
    for i in tqdm(range(len(lines))):
        content = lines[i]
          
        new_line = tokenize_by_bpe_model(content)
        f_out.write(new_line.upper())
        f_out.write("\n")
    f_out.close()

if __name__ == "__main__":
    main()
