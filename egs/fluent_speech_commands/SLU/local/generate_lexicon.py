import argparse

import pandas
from tqdm import tqdm


def generate_lexicon(corpus_dir, lm_dir):
    data = pandas.read_csv(
        str(corpus_dir) + "/data/train_data.csv", index_col=0, header=0
    )
    vocab_transcript = set()
    vocab_frames = set()
    transcripts = data["transcription"].tolist()
    frames = list(
        i
        for i in zip(
            data["action"].tolist(), data["object"].tolist(), data["location"].tolist()
        )
    )

    for transcript in tqdm(transcripts):
        for word in transcript.split():
            vocab_transcript.add(word)

    for frame in tqdm(frames):
        for word in frame:
            vocab_frames.add("_".join(word.split()))

    with open(lm_dir + "/words_transcript.txt", "w") as lexicon_transcript_file:
        lexicon_transcript_file.write("<UNK> 1" + "\n")
        lexicon_transcript_file.write("<s> 2" + "\n")
        lexicon_transcript_file.write("</s> 0" + "\n")
        id = 3
        for vocab in vocab_transcript:
            lexicon_transcript_file.write(vocab + " " + str(id) + "\n")
            id += 1

    with open(lm_dir + "/words_frames.txt", "w") as lexicon_frames_file:
        lexicon_frames_file.write("<UNK> 1" + "\n")
        lexicon_frames_file.write("<s> 2" + "\n")
        lexicon_frames_file.write("</s> 0" + "\n")
        id = 3
        for vocab in vocab_frames:
            lexicon_frames_file.write(vocab + " " + str(id) + "\n")
            id += 1


parser = argparse.ArgumentParser()
parser.add_argument("corpus_dir")
parser.add_argument("lm_dir")


def main():
    args = parser.parse_args()

    generate_lexicon(args.corpus_dir, args.lm_dir)


main()
