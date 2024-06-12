Kaldi-based forced alignment
============================

This section describes in detail how to use `kaldi-decoder`_
for **FST-based** ``forced alignment`` with models trained by `CTC`_ loss.

.. hint::

  We have a colab notebook walking you through this section step by step.

  |kaldi-based forced alignment colab notebook|

  .. |kaldi-based forced alignment colab notebook| image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://github.com/k2-fsa/colab/blob/master/icefall/ctc_forced_alignment_fst_based_kaldi.ipynb

Prepare the environment
-----------------------

Before you continue, make sure you have setup `icefall`_ by following :ref:`install icefall`.

.. hint::

   You don't need to install `Kaldi`_. We will ``NOT`` use `Kaldi`_ below.

Get the test data
-----------------

We use the test wave
from `CTC FORCED ALIGNMENT API TUTORIAL <https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html>`_

.. code-block:: python3

  import torchaudio

  # Download test wave
  speech_file = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
  print(speech_file)
  waveform, sr = torchaudio.load(speech_file)
  transcript = "i had that curiosity beside me at this moment".split()
  print(waveform.shape, sr)

  assert waveform.ndim == 2
  assert waveform.shape[0] == 1
  assert sr == 16000

The test wave is downloaded to::

  $HOME/.cache/torch/hub/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav</td>
      <td>
       <audio title="Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        i had that curiosity beside me at this moment
      </td>
    </tr>
  </table>

We use the test model
from `CTC FORCED ALIGNMENT API TUTORIAL <https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html>`_

.. code-block:: python3

  import torch

  bundle = torchaudio.pipelines.MMS_FA

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = bundle.get_model(with_star=False).to(device)

The model is downloaded to::

  $HOME/.cache/torch/hub/checkpoints/model.pt

Compute log_probs
-----------------

.. code-block:: bash

  with torch.inference_mode():
      emission, _ = model(waveform.to(device))
      print(emission.shape)

It should print::

  torch.Size([1, 169, 28])

Create token2id and id2token
----------------------------

.. code-block:: python3

    token2id = bundle.get_dict(star=None)
    id2token = {i:t for t, i in token2id.items()}
    token2id["<eps>"] = 0
    del token2id["-"]

Create word2id and id2word
--------------------------

.. code-block:: python3

  words = list(set(transcript))
  word2id = dict()
  word2id['eps'] = 0
  for i, w in enumerate(words):
    word2id[w] = i + 1

  id2word = {i:w for w, i in word2id.items()}

Note that we only use words from the transcript of the test wave.

Generate lexicon-related files
------------------------------

We use the code below to generate the following 4 files:

  - ``lexicon.txt``
  - ``tokens.txt``
  - ``words.txt``
  - ``lexicon_disambig.txt``

.. caution::

   ``words.txt`` contains only words from the transcript of the test wave.

.. code-block:: python3

  from prepare_lang import add_disambig_symbols

  lexicon = [(w, list(w)) for w in word2id if w != "eps"]
  lexicon_disambig, max_disambig_id = add_disambig_symbols(lexicon)

  with open('lexicon.txt', 'w', encoding='utf-8') as f:
    for w, tokens in lexicon:
      f.write(f"{w} {' '.join(tokens)}\n")

  with open('lexicon_disambig.txt', 'w', encoding='utf-8') as f:
    for w, tokens in lexicon_disambig:
      f.write(f"{w} {' '.join(tokens)}\n")

  with open('tokens.txt', 'w', encoding='utf-8') as f:
    for t, i in token2id.items():
      if t == '-':
        t = "<eps>"
      f.write(f"{t} {i}\n")

    for k in range(max_disambig_id + 2):
      f.write(f"#{k} {len(token2id) + k}\n")

  with open('words.txt', 'w', encoding='utf-8') as f:
    for w, i in word2id.items():
      f.write(f"{w} {i}\n")
    f.write(f'#0 {len(word2id)}\n')


To give you an idea about what the generated files look like::

  head -n 50 lexicon.txt lexicon_disambig.txt tokens.txt words.txt

prints::

  ==> lexicon.txt <==
  moment m o m e n t
  beside b e s i d e
  i i
  this t h i s
  curiosity c u r i o s i t y
  had h a d
  that t h a t
  at a t
  me m e

  ==> lexicon_disambig.txt <==
  moment m o m e n t
  beside b e s i d e
  i i
  this t h i s
  curiosity c u r i o s i t y
  had h a d
  that t h a t
  at a t
  me m e

  ==> tokens.txt <==
  a 1
  i 2
  e 3
  n 4
  o 5
  u 6
  t 7
  s 8
  r 9
  m 10
  k 11
  l 12
  d 13
  g 14
  h 15
  y 16
  b 17
  p 18
  w 19
  c 20
  v 21
  j 22
  z 23
  f 24
  ' 25
  q 26
  x 27
  <eps> 0
  #0 28
  #1 29

  ==> words.txt <==
  eps 0
  moment 1
  beside 2
  i 3
  this 4
  curiosity 5
  had 6
  that 7
  at 8
  me 9
  #0 10

.. note::

   This test model uses characters as modeling unit. If you use other types of
   modeling unit, the same code can be used without any change.

Convert transcript to an FST graph
----------------------------------

.. code-block:: bash

   egs/librispeech/ASR/local/prepare_lang_fst.py --lang-dir ./

The above command should generate two files ``H.fst`` and ``HL.fst``. We will
use ``HL.fst`` below::

  -rw-r--r-- 1 root root  13K Jun 12 08:28 H.fst
  -rw-r--r-- 1 root root 3.7K Jun 12 08:28 HL.fst

Force aligner
-------------

Now, everything is ready. We can use the following code to get forced alignments.

.. code-block:: python3

  from kaldi_decoder import DecodableCtc, FasterDecoder, FasterDecoderOptions
  import kaldifst

  def force_align():
      HL = kaldifst.StdVectorFst.read("./HL.fst")
      decodable = DecodableCtc(emission[0].contiguous().cpu().numpy())
      decoder_opts = FasterDecoderOptions(max_active=3000)
      decoder = FasterDecoder(HL, decoder_opts)
      decoder.decode(decodable)
      if not decoder.reached_final():
          print(f"failed to decode xxx")
          return None
      ok, best_path = decoder.get_best_path()

      (
          ok,
          isymbols_out,
          osymbols_out,
          total_weight,
      ) = kaldifst.get_linear_symbol_sequence(best_path)
      if not ok:
          print(f"failed to get linear symbol sequence for xxx")
          return None

      # We need to use i-1 here since we have incremented tokens during
      # HL construction
      alignment = [i-1 for i in isymbols_out]
      return alignment

  alignment = force_align()

  for i, a in enumerate(alignment):
    print(i, id2token[a])

The output should be identical to
`<https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html#frame-level-alignments>`_.

For ease of reference, we list the output below::

  0 -
  1 -
  2 -
  3 -
  4 -
  5 -
  6 -
  7 -
  8 -
  9 -
  10 -
  11 -
  12 -
  13 -
  14 -
  15 -
  16 -
  17 -
  18 -
  19 -
  20 -
  21 -
  22 -
  23 -
  24 -
  25 -
  26 -
  27 -
  28 -
  29 -
  30 -
  31 -
  32 i
  33 -
  34 -
  35 h
  36 h
  37 a
  38 -
  39 -
  40 -
  41 d
  42 -
  43 -
  44 t
  45 h
  46 -
  47 a
  48 -
  49 -
  50 t
  51 -
  52 -
  53 -
  54 c
  55 -
  56 -
  57 -
  58 u
  59 u
  60 -
  61 -
  62 -
  63 r
  64 -
  65 i
  66 -
  67 -
  68 -
  69 -
  70 -
  71 -
  72 o
  73 -
  74 -
  75 -
  76 -
  77 -
  78 -
  79 s
  80 -
  81 -
  82 -
  83 i
  84 -
  85 t
  86 -
  87 -
  88 y
  89 -
  90 -
  91 -
  92 -
  93 b
  94 -
  95 e
  96 -
  97 -
  98 -
  99 -
  100 -
  101 s
  102 -
  103 -
  104 -
  105 -
  106 -
  107 -
  108 -
  109 -
  110 i
  111 -
  112 -
  113 d
  114 e
  115 -
  116 m
  117 -
  118 -
  119 e
  120 -
  121 -
  122 -
  123 -
  124 a
  125 -
  126 -
  127 t
  128 -
  129 t
  130 h
  131 -
  132 i
  133 -
  134 -
  135 -
  136 s
  137 -
  138 -
  139 -
  140 -
  141 m
  142 -
  143 -
  144 o
  145 -
  146 -
  147 -
  148 m
  149 -
  150 -
  151 e
  152 -
  153 n
  154 -
  155 t
  156 -
  157 -
  158 -
  159 -
  160 -
  161 -
  162 -
  163 -
  164 -
  165 -
  166 -
  167 -
  168 -

To merge tokens, we use::

  from icefall.ctc import merge_tokens
  token_spans = merge_tokens(alignment)
  for span in token_spans:
    print(id2token[span.token], span.start, span.end)

The output is given below::

  i 32 33
  h 35 37
  a 37 38
  d 41 42
  t 44 45
  h 45 46
  a 47 48
  t 50 51
  c 54 55
  u 58 60
  r 63 64
  i 65 66
  o 72 73
  s 79 80
  i 83 84
  t 85 86
  y 88 89
  b 93 94
  e 95 96
  s 101 102
  i 110 111
  d 113 114
  e 114 115
  m 116 117
  e 119 120
  a 124 125
  t 127 128
  t 129 130
  h 130 131
  i 132 133
  s 136 137
  m 141 142
  o 144 145
  m 148 149
  e 151 152
  n 153 154
  t 155 156

All of the code below is copied and modified
from `<https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html>`_.

Segment each word using the computed alignments
-----------------------------------------------

.. code-block:: python3

  def unflatten(list_, lengths):
      assert len(list_) == sum(lengths)
      i = 0
      ret = []
      for l in lengths:
          ret.append(list_[i : i + l])
          i += l
      return ret


  word_spans = unflatten(token_spans, [len(word) for word in transcript])
  print(word_spans)

The output is::

  [[TokenSpan(token=2, start=32, end=33)],
   [TokenSpan(token=15, start=35, end=37), TokenSpan(token=1, start=37, end=38), TokenSpan(token=13, start=41, end=42)],
   [TokenSpan(token=7, start=44, end=45), TokenSpan(token=15, start=45, end=46), TokenSpan(token=1, start=47, end=48), TokenSpan(token=7, start=50, end=51)],
   [TokenSpan(token=20, start=54, end=55), TokenSpan(token=6, start=58, end=60), TokenSpan(token=9, start=63, end=64), TokenSpan(token=2, start=65, end=66), TokenSpan(token=5, start=72, end=73), TokenSpan(token=8, start=79, end=80), TokenSpan(token=2, start=83, end=84), TokenSpan(token=7, start=85, end=86), TokenSpan(token=16, start=88, end=89)],
   [TokenSpan(token=17, start=93, end=94), TokenSpan(token=3, start=95, end=96), TokenSpan(token=8, start=101, end=102), TokenSpan(token=2, start=110, end=111), TokenSpan(token=13, start=113, end=114), TokenSpan(token=3, start=114, end=115)],
   [TokenSpan(token=10, start=116, end=117), TokenSpan(token=3, start=119, end=120)],
   [TokenSpan(token=1, start=124, end=125), TokenSpan(token=7, start=127, end=128)],
   [TokenSpan(token=7, start=129, end=130), TokenSpan(token=15, start=130, end=131), TokenSpan(token=2, start=132, end=133), TokenSpan(token=8, start=136, end=137)],
   [TokenSpan(token=10, start=141, end=142), TokenSpan(token=5, start=144, end=145), TokenSpan(token=10, start=148, end=149), TokenSpan(token=3, start=151, end=152), TokenSpan(token=4, start=153, end=154), TokenSpan(token=7, start=155, end=156)]
  ]


.. code-block:: python3

  def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
      ratio = waveform.size(1) / num_frames
      x0 = int(ratio * spans[0].start)
      x1 = int(ratio * spans[-1].end)
      print(f"{transcript} {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
      segment = waveform[:, x0:x1]
      return IPython.display.Audio(segment.numpy(), rate=sample_rate)
  num_frames = emission.size(1)

.. code-block:: python3

   preview_word(waveform, word_spans[0], num_frames, transcript[0])
   preview_word(waveform, word_spans[1], num_frames, transcript[1])
   preview_word(waveform, word_spans[2], num_frames, transcript[2])
   preview_word(waveform, word_spans[3], num_frames, transcript[3])
   preview_word(waveform, word_spans[4], num_frames, transcript[4])
   preview_word(waveform, word_spans[5], num_frames, transcript[5])
   preview_word(waveform, word_spans[6], num_frames, transcript[6])
   preview_word(waveform, word_spans[7], num_frames, transcript[7])
   preview_word(waveform, word_spans[8], num_frames, transcript[8])

The segmented wave of each word along with its time stamp is given below:

.. raw:: html

  <table>
    <tr>
      <th>Word</th>
      <th>Time</th>
      <th>Wave</th>
    </tr>
    <tr>
      <td>i</td>
      <td>0.644 - 0.664 sec</td>
      <td>
       <audio title="i.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/i.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>had</td>
      <td>0.704 - 0.845 sec</td>
      <td>
       <audio title="had.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/had.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>that</td>
      <td>0.885 - 1.026 sec</td>
      <td>
       <audio title="that.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/that.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>curiosity</td>
      <td>1.086 - 1.790 sec</td>
      <td>
       <audio title="curiosity.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/curiosity.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>beside</td>
      <td>1.871 - 2.314 sec</td>
      <td>
       <audio title="beside.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/beside.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>me</td>
      <td>2.334 - 2.414 sec</td>
      <td>
       <audio title="me.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/me.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>at</td>
      <td>2.495 - 2.575 sec</td>
      <td>
       <audio title="at.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/at.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>this</td>
      <td>2.595 - 2.756 sec</td>
      <td>
       <audio title="this.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/this.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
    <tr>
      <td>moment</td>
      <td>2.837 - 3.138 sec</td>
      <td>
       <audio title="moment.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/moment.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>

We repost the whole wave below for ease of reference:

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav</td>
      <td>
       <audio title="Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav" controls="controls">
             <source src="/icefall/_static/kaldi-align/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        i had that curiosity beside me at this moment
      </td>
    </tr>
  </table>

Summary
-------

Congratulations! You have succeeded in using the FST-based approach to
compute alignment of a test wave.
