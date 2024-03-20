#!/usr/bin/env python3
#
# Copyright      2023 Xiaomi Corporation     (Author: Zengwei Yao)
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
This script exports a VITS model from PyTorch to ONNX.

Export the model to ONNX:
./vits/export-onnx.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt

It will generate one file inside vits/exp:
  - vits-epoch-1000.onnx

See ./test_onnx.py for how to use the exported ONNX models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import onnx
import torch
import torch.nn as nn
from tokenizer import Tokenizer
from train import get_model, get_params

from icefall.checkpoint import load_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="vits/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="""If not empty, valid values are: low, medium, high.
        It controls the model size. low -> runs faster.
        """,
    )

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class OnnxModel(nn.Module):
    """A wrapper for VITS generator."""

    def __init__(self, model: nn.Module):
        """
        Args:
          model:
            A VITS generator.
          frame_shift:
            The frame shift in samples.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_lens: torch.Tensor,
        noise_scale: float = 0.667,
        alpha: float = 1.0,
        noise_scale_dur: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Please see the help information of VITS.inference_batch

        Args:
          tokens:
            Input text token indexes (1, T_text)
          tokens_lens:
            Number of tokens of shape (1,)
          noise_scale (float):
            Noise scale parameter for flow.
          noise_scale_dur (float):
            Noise scale parameter for duration predictor.
          alpha (float):
            Alpha parameter to control the speed of generated speech.

        Returns:
          Return a tuple containing:
            - audio, generated wavform tensor, (B, T_wav)
        """
        audio, _, _ = self.model.generator.inference(
            text=tokens,
            text_lengths=tokens_lens,
            noise_scale=noise_scale,
            noise_scale_dur=noise_scale_dur,
            alpha=alpha,
        )
        return audio


def export_model_onnx(
    model: nn.Module,
    model_filename: str,
    vocab_size: int,
    opset_version: int = 11,
) -> None:
    """Export the given generator model to ONNX format.
    The exported model has one input:

        - tokens, a tensor of shape (1, T_text); dtype is torch.int64

    and it has one output:

        - audio, a tensor of shape (1, T'); dtype is torch.float32

    Args:
      model:
        The VITS generator.
      model_filename:
        The filename to save the exported ONNX model.
      vocab_size:
        Number of tokens used in training.
      opset_version:
        The opset version to use.
    """
    tokens = torch.randint(low=0, high=vocab_size, size=(1, 13), dtype=torch.int64)
    tokens_lens = torch.tensor([tokens.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_dur = torch.tensor([1], dtype=torch.float32)
    alpha = torch.tensor([1], dtype=torch.float32)

    torch.onnx.export(
        model,
        (tokens, tokens_lens, noise_scale, alpha, noise_scale_dur),
        model_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "tokens_lens",
            "noise_scale",
            "alpha",
            "noise_scale_dur",
        ],
        output_names=["audio"],
        dynamic_axes={
            "tokens": {0: "N", 1: "T"},
            "tokens_lens": {0: "N"},
            "audio": {0: "N", 1: "T"},
        },
    )

    if model.model.spks is None:
        num_speakers = 1
    else:
        num_speakers = model.model.spks

    meta_data = {
        "model_type": "vits",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "icefall",  # must be icefall for models from icefall
        "language": "English",
        "voice": "en-us",  # Choose your language appropriately
        "has_espeak": 1,
        "n_speakers": num_speakers,
        "sample_rate": model.model.sampling_rate,  # Must match the real sample rate
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=model_filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    tokenizer = Tokenizer(params.tokens)
    params.blank_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    model.to("cpu")
    model.eval()

    model = OnnxModel(model=model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"generator parameters: {num_param}, or {num_param/1000/1000} M")

    suffix = f"epoch-{params.epoch}"

    opset_version = 13

    logging.info("Exporting encoder")
    model_filename = params.exp_dir / f"vits-{suffix}.onnx"
    export_model_onnx(
        model,
        model_filename,
        params.vocab_size,
        opset_version=opset_version,
    )
    logging.info(f"Exported generator to {model_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

"""
Supported languages.

LJSpeech is using "en-us" from the second column.

Pty Language       Age/Gender VoiceName          File                 Other Languages
 5  af              --/M      Afrikaans          gmw/af
 5  am              --/M      Amharic            sem/am
 5  an              --/M      Aragonese          roa/an
 5  ar              --/M      Arabic             sem/ar
 5  as              --/M      Assamese           inc/as
 5  az              --/M      Azerbaijani        trk/az
 5  ba              --/M      Bashkir            trk/ba
 5  be              --/M      Belarusian         zle/be
 5  bg              --/M      Bulgarian          zls/bg
 5  bn              --/M      Bengali            inc/bn
 5  bpy             --/M      Bishnupriya_Manipuri inc/bpy
 5  bs              --/M      Bosnian            zls/bs
 5  ca              --/M      Catalan            roa/ca
 5  chr-US-Qaaa-x-west --/M      Cherokee_          iro/chr
 5  cmn             --/M      Chinese_(Mandarin,_latin_as_English) sit/cmn              (zh-cmn 5)(zh 5)
 5  cmn-latn-pinyin --/M      Chinese_(Mandarin,_latin_as_Pinyin) sit/cmn-Latn-pinyin  (zh-cmn 5)(zh 5)
 5  cs              --/M      Czech              zlw/cs
 5  cv              --/M      Chuvash            trk/cv
 5  cy              --/M      Welsh              cel/cy
 5  da              --/M      Danish             gmq/da
 5  de              --/M      German             gmw/de
 5  el              --/M      Greek              grk/el
 5  en-029          --/M      English_(Caribbean) gmw/en-029           (en 10)
 2  en-gb           --/M      English_(Great_Britain) gmw/en               (en 2)
 5  en-gb-scotland  --/M      English_(Scotland) gmw/en-GB-scotland   (en 4)
 5  en-gb-x-gbclan  --/M      English_(Lancaster) gmw/en-GB-x-gbclan   (en-gb 3)(en 5)
 5  en-gb-x-gbcwmd  --/M      English_(West_Midlands) gmw/en-GB-x-gbcwmd   (en-gb 9)(en 9)
 5  en-gb-x-rp      --/M      English_(Received_Pronunciation) gmw/en-GB-x-rp       (en-gb 4)(en 5)
 2  en-us           --/M      English_(America)  gmw/en-US            (en 3)
 5  en-us-nyc       --/M      English_(America,_New_York_City) gmw/en-US-nyc
 5  eo              --/M      Esperanto          art/eo
 5  es              --/M      Spanish_(Spain)    roa/es
 5  es-419          --/M      Spanish_(Latin_America) roa/es-419           (es-mx 6)
 5  et              --/M      Estonian           urj/et
 5  eu              --/M      Basque             eu
 5  fa              --/M      Persian            ira/fa
 5  fa-latn         --/M      Persian_(Pinglish) ira/fa-Latn
 5  fi              --/M      Finnish            urj/fi
 5  fr-be           --/M      French_(Belgium)   roa/fr-BE            (fr 8)
 5  fr-ch           --/M      French_(Switzerland) roa/fr-CH            (fr 8)
 5  fr-fr           --/M      French_(France)    roa/fr               (fr 5)
 5  ga              --/M      Gaelic_(Irish)     cel/ga
 5  gd              --/M      Gaelic_(Scottish)  cel/gd
 5  gn              --/M      Guarani            sai/gn
 5  grc             --/M      Greek_(Ancient)    grk/grc
 5  gu              --/M      Gujarati           inc/gu
 5  hak             --/M      Hakka_Chinese      sit/hak
 5  haw             --/M      Hawaiian           map/haw
 5  he              --/M      Hebrew             sem/he
 5  hi              --/M      Hindi              inc/hi
 5  hr              --/M      Croatian           zls/hr               (hbs 5)
 5  ht              --/M      Haitian_Creole     roa/ht
 5  hu              --/M      Hungarian          urj/hu
 5  hy              --/M      Armenian_(East_Armenia) ine/hy               (hy-arevela 5)
 5  hyw             --/M      Armenian_(West_Armenia) ine/hyw              (hy-arevmda 5)(hy 8)
 5  ia              --/M      Interlingua        art/ia
 5  id              --/M      Indonesian         poz/id
 5  io              --/M      Ido                art/io
 5  is              --/M      Icelandic          gmq/is
 5  it              --/M      Italian            roa/it
 5  ja              --/M      Japanese           jpx/ja
 5  jbo             --/M      Lojban             art/jbo
 5  ka              --/M      Georgian           ccs/ka
 5  kk              --/M      Kazakh             trk/kk
 5  kl              --/M      Greenlandic        esx/kl
 5  kn              --/M      Kannada            dra/kn
 5  ko              --/M      Korean             ko
 5  kok             --/M      Konkani            inc/kok
 5  ku              --/M      Kurdish            ira/ku
 5  ky              --/M      Kyrgyz             trk/ky
 5  la              --/M      Latin              itc/la
 5  lb              --/M      Luxembourgish      gmw/lb
 5  lfn             --/M      Lingua_Franca_Nova art/lfn
 5  lt              --/M      Lithuanian         bat/lt
 5  ltg             --/M      Latgalian          bat/ltg
 5  lv              --/M      Latvian            bat/lv
 5  mi              --/M      Māori             poz/mi
 5  mk              --/M      Macedonian         zls/mk
 5  ml              --/M      Malayalam          dra/ml
 5  mr              --/M      Marathi            inc/mr
 5  ms              --/M      Malay              poz/ms
 5  mt              --/M      Maltese            sem/mt
 5  mto             --/M      Totontepec_Mixe    miz/mto
 5  my              --/M      Myanmar_(Burmese)  sit/my
 5  nb              --/M      Norwegian_Bokmål  gmq/nb               (no 5)
 5  nci             --/M      Nahuatl_(Classical) azc/nci
 5  ne              --/M      Nepali             inc/ne
 5  nl              --/M      Dutch              gmw/nl
 5  nog             --/M      Nogai              trk/nog
 5  om              --/M      Oromo              cus/om
 5  or              --/M      Oriya              inc/or
 5  pa              --/M      Punjabi            inc/pa
 5  pap             --/M      Papiamento         roa/pap
 5  piqd            --/M      Klingon            art/piqd
 5  pl              --/M      Polish             zlw/pl
 5  pt              --/M      Portuguese_(Portugal) roa/pt               (pt-pt 5)
 5  pt-br           --/M      Portuguese_(Brazil) roa/pt-BR            (pt 6)
 5  py              --/M      Pyash              art/py
 5  qdb             --/M      Lang_Belta         art/qdb
 5  qu              --/M      Quechua            qu
 5  quc             --/M      K'iche'            myn/quc
 5  qya             --/M      Quenya             art/qya
 5  ro              --/M      Romanian           roa/ro
 5  ru              --/M      Russian            zle/ru
 5  ru-cl           --/M      Russian_(Classic)  zle/ru-cl
 2  ru-lv           --/M      Russian_(Latvia)   zle/ru-LV
 5  sd              --/M      Sindhi             inc/sd
 5  shn             --/M      Shan_(Tai_Yai)     tai/shn
 5  si              --/M      Sinhala            inc/si
 5  sjn             --/M      Sindarin           art/sjn
 5  sk              --/M      Slovak             zlw/sk
 5  sl              --/M      Slovenian          zls/sl
 5  smj             --/M      Lule_Saami         urj/smj
 5  sq              --/M      Albanian           ine/sq
 5  sr              --/M      Serbian            zls/sr
 5  sv              --/M      Swedish            gmq/sv
 5  sw              --/M      Swahili            bnt/sw
 5  ta              --/M      Tamil              dra/ta
 5  te              --/M      Telugu             dra/te
 5  th              --/M      Thai               tai/th
 5  tk              --/M      Turkmen            trk/tk
 5  tn              --/M      Setswana           bnt/tn
 5  tr              --/M      Turkish            trk/tr
 5  tt              --/M      Tatar              trk/tt
 5  ug              --/M      Uyghur             trk/ug
 5  uk              --/M      Ukrainian          zle/uk
 5  ur              --/M      Urdu               inc/ur
 5  uz              --/M      Uzbek              trk/uz
 5  vi              --/M      Vietnamese_(Northern) aav/vi
 5  vi-vn-x-central --/M      Vietnamese_(Central) aav/vi-VN-x-central
 5  vi-vn-x-south   --/M      Vietnamese_(Southern) aav/vi-VN-x-south
 5  yue             --/M      Chinese_(Cantonese) sit/yue              (zh-yue 5)(zh 8)
 5  yue             --/M      Chinese_(Cantonese,_latin_as_Jyutping) sit/yue-Latn-jyutping (zh-yue 5)(zh 8)
"""
