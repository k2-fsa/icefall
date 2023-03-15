import argparse
import logging
import torch
import k2

from icefall.lexicon import Lexicon
from icefall.char_graph_compiler import CharCtcTrainingGraphCompiler

parser = argparse.ArgumentParser()

parser.add_argument(
    "--transcript",
    type=str,
    default="download/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt",
    help="Original transcript of aishell dataset"
)

parser.add_argument(
    "--lang-dir",
    type=str,
    default="data/lang_char/",
    help="A folder containing tokens.txt"
)

parser.add_argument(
    "--lm-data",
    type=str,
    default="lm-data.pt",
    help="Path of the output lm data."
)

def prepare_training_transcript(args):
    transcript = args.transcript
    lang_dir = args.lang_dir
    lm_data = args.lm_data
    
    device = torch.device("cpu")
    lexicon = Lexicon(lang_dir)
    graph_compiler = CharCtcTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
        oov="<unk>",
    )
    
    # char2index is a dictionary from words to integer ids.  No need to reserve
    # space for epsilon, etc.; the chars are just used as a convenient way to
    # compress the sequences of char.
    word2index = dict()
    #print(lexicon.tokens)
        
    
    char_table = []  # Will be a list-of-list-of-int, representing char.
    
    step = 5000
    processed = 0
    sentences = []
    with open(transcript, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            if processed % step == 0 and processed:
                logging.info(
                    f"Processed number of lines: {processed} "
                )
            processed += 1
            
            uid, sentence = line.split(" ", 1)
            sentence = sentence.strip().replace(" ", "") # remove the blanks
            #y = graph_compiler.texts_to_ids(sentence) # [[0], [1], [2]]
            #y = [char[0] for char in y]
            
            #import pdb; pdb.set_trace()
            for char in sentence:
                if char not in word2index:
                    char_id = graph_compiler.texts_to_ids(char)
                    word2index[char] = len(char_table)
                    char_table.append(char_id[0])
            #import pdb; pdb.set_trace()
            sentences.append([word2index[ch] for ch in sentence])
    
    chars = k2.ragged.RaggedTensor(char_table)
    sentences = k2.ragged.RaggedTensor(sentences)
    
    output = dict(words=chars, sentences=sentences)
    
    num_sentences = sentences.dim0
    logging.info(f"Computing sentence lengths, num_sentences: {num_sentences}")
    sentence_lengths = [0] * num_sentences
    for i in range(num_sentences):
        if step and i % step == 0:
            logging.info(
                f"Processed number of lines: {i} ({i/num_sentences*100: .3f}%)"
            )

        word_ids = sentences[i]

        # NOTE: If word_ids is a tensor with only 1 entry,
        # token_ids is a torch.Tensor
        token_ids = chars[word_ids]
        if isinstance(token_ids, k2.RaggedTensor):
            token_ids = token_ids.values

        # token_ids is a 1-D tensor containing the BPE tokens
        # of the current sentence

        sentence_lengths[i] = token_ids.numel()
    
    #output["sentence_lengths"] = torch.tensor(sentence_lengths, dtype=torch.int32)
    
    sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.int32)
    
    data = {}
    
    indices = torch.argsort(sentence_lengths, descending=True)
    
    sorted_sentences = sentences[indices.to(torch.int32)]
    sorted_sentence_lengths = sentence_lengths[indices]
    
    data["sentences"] = sorted_sentences
    data["sentence_lengths"] = sorted_sentence_lengths
    data["words"] = output["words"]
    
    # store the lm data
    torch.save(data, lm_data)
    logging.info(f"Saved to {lm_data}")

if __name__ == "__main__":
    args = parser.parse_args()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    prepare_training_transcript(args)