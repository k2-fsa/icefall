import torch
import torch.nn as nn
from fairseq2.nn.embedding import Embedding
from seamless_communication.models.inference import Translator
from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYModel,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from fairseq2.generation import (
    Seq2SeqGenerator,
    SequenceGeneratorOptions,
    SequenceGeneratorOutput,
    SequenceToTextGenerator,
    SequenceToTextOutput,
)
from seamless_communication.models.unity.model import UnitYModel, UnitYX2TModel

import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
audio_file="/mnt/samsung-t7/yuekai/asr/Triton-ASR-Client/datasets/mini_en/wav/1089-134686-0001.wav"
src_lang="cmn"

audio_file="/mnt/samsung-t7/yuekai/asr/Triton-ASR-Client/datasets/mini_zh/wav/long.wav"
src_lang="eng"
target_lang = "cmn"

audio_input = torchaudio.load(audio_file)[0]
feature = ta_kaldi.fbank(audio_input, num_mel_bins=80)
# feature shape is (T, F), convert it to (B, T, F), source_seq_lens tracks T 
source_seqs = feature.unsqueeze(0)
source_seq_lens = torch.tensor([feature.shape[0]])

# Initialize a Translator object with a multitask model, vocoder on the GPU.


# translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:2"), dtype=torch.float16)

# transcribed_text, _, _ = translator.predict(audio_file, "asr", src_lang)

# print(transcribed_text)


model_name_or_card = "seamlessM4T_medium"
device = torch.device("cuda:3")

# cast source_seq_lens, source_seqs to device, dtype to torch.float16
source_seq_lens = source_seq_lens.to(device=device, dtype=torch.float16)
source_seqs = source_seqs.to(device=device, dtype=torch.float16)



dtype = torch.float16
model = load_unity_model(model_name_or_card, device=device, dtype=dtype)
model.eval()
model.text_decoder_frontend.embed = Embedding(num_embeddings=6257, embedding_dim=1024 ,pad_idx=0, scaled=True)
model.final_proj = nn.Linear(1024, 6257)
model.half()
print(model.text_decoder_frontend.embed, model.text_encoder_frontend.embed.weight.dtype, type(model.text_encoder_frontend.embed), type(model.text_encoder_frontend.embed.weight))
print(model.final_proj, model.final_proj.weight.dtype, type(model.final_proj), type(model.final_proj.weight))
#input()
exit(0)
text_tokenizer = load_unity_text_tokenizer(model_name_or_card)
#print(text_tokenizer.model.eos_idx, text_tokenizer.model.pad_idx)
#text_tokenizer_encoder = text_tokenizer.create_encoder(lang=target_lang, mode="target")
#text_tokenizer_decoder = text_tokenizer.create_decoder()
# print attritbut of text_tokenizer_encoder
#print(text_tokenizer.vocab_info)
#print(text_tokenizer_encoder("其中广州深圳甚至出现了多个日光盘"))
#print(text_tokenizer_decoder(torch.tensor([3,256200,137139,252603,250476,250590,1,84778,148897,249568,249352,249947,249050,250520,254508])))

# store all vocab in a file
# with open("vocab.txt", "w") as f:
#     for i in range(256206):
#         f.write(f"{i}: " + text_tokenizer_decoder(torch.tensor([i]))[0].bytes().decode("utf-8")+ "\n")
#     f.close()
# exit(0)



# def decode(
#     self,
#     seqs: Tensor,
#     seq_lens: Optional[Tensor],
#     encoder_output: Tensor,
#     encoder_padding_mask: Optional[Tensor],
#     state_bag: Optional[IncrementalStateBag] = None,
# ) -> Tuple[Tensor, Optional[Tensor]]:
#     seqs, padding_mask = self.text_decoder_frontend(seqs, seq_lens, state_bag)

#     return self.text_decoder(  # type: ignore[no-any-return]
#         seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
#     )

# def decoding(model, feature):
#     seqs, padding_mask = model.speech_encoder_frontend(seqs, seq_lens)
#     speech_encoder(seqs, padding_mask)

#     decoder_output, decoder_padding_mask = self.decode(
#         batch.target_seqs,
#         batch.target_seq_lens,
#         encoder_output,
#         encoder_padding_mask,
#     )

#     text_logits = model.final_project(decoder_output, decoder_padding_mask)

text_max_len_a = 1
text_max_len_b = 200

text_opts = SequenceGeneratorOptions(
    beam_size=5, soft_max_seq_len=(text_max_len_a, text_max_len_b)
)

s2t_model = UnitYX2TModel(
    encoder_frontend=model.speech_encoder_frontend,
    encoder=model.speech_encoder,
    decoder_frontend=model.text_decoder_frontend,
    decoder=model.text_decoder,
    final_proj=model.final_proj,
    pad_idx=model.pad_idx,
)
s2t_generator = SequenceToTextGenerator(
    s2t_model, text_tokenizer, target_lang, text_opts
)

text_output = s2t_generator.generate_ex(source_seqs, source_seq_lens)
print(text_output.generator_output.results[0][0].seq.cpu().tolist())
# sentence = text_output.sentences[0]
# print(sentence, type(sentence))
# sentence = sentence.bytes().decode("utf-8")
