from typing import Dict
import ast
from lhotse import load_manifest, load_manifest_lazy
from lhotse.cut import Cut, CutSet
from text_normalization import remove_non_alphabetic
from tqdm import tqdm
import os

def get_facebook_biasing_list(
    test_set: str,
    use_distractors: bool = False,
    num_distractors: int = 100,
) -> Dict:
    assert num_distractors in (100,500,1000,2000), num_distractors
    if test_set == "test-clean":
        biasing_file = f"data/context_biasing/fbai-speech/is21_deep_bias/ref/test-clean.biasing_{num_distractors}.tsv"
    elif test_set == "test-other":
        biasing_file = f"data/context_biasing/fbai-speech/is21_deep_bias/ref/test-other.biasing_{num_distractors}.tsv"
    else:
        raise ValueError(f"Unseen test set {test_set}")
    
    f = open(biasing_file, 'r')
    data = f.readlines()
    f.close()
    
    output = dict()
    for line in data:
        id, _, l1, l2 = line.split('\t')
        if use_distractors:
            biasing_list = ast.literal_eval(l2)
        else:
            biasing_list = ast.literal_eval(l1)
        biasing_list = [w.strip().upper() for w in biasing_list] 
        output[id] = " ".join(biasing_list)
         
    return output

def get_rare_words(subset: str, min_count: int):
    txt_path = f"data/tmp/transcript_words_{subset}.txt"
    rare_word_file = f"data/context_biasing/{subset}_rare_words_{min_count}.txt"
    
    if os.path.exists(rare_word_file):
        print("File exists, do not proceed!")
        return
    print(f"Finding rare words in the manifest.")
    count_file = f"data/tmp/transcript_words_{subset}_count.txt"
    if not os.path.exists(count_file):
        with open(txt_path, "r") as file:
            words = file.read().upper().split()
            word_count = {}
            for word in words:
                word = remove_non_alphabetic(word, strict=False)
                word = word.split()
                for w in word:
                    if w not in word_count:
                        word_count[w] = 1
                    else:
                        word_count[w] += 1
        
        with open(count_file, 'w') as fout:
            for w in word_count:
                fout.write(f"{w}\t{word_count[w]}")
    else:
        word_count = {}
        with open(count_file, 'r') as fin:
            word_count = fin.read().split('\n')
            word_count = [pair.split() for pair in word_count]

    print(f"A total of {len(word_count)} words appeared!")
    rare_words = []
    for k in word_count:
        if int(word_count[k]) <= min_count:
            rare_words.append(k+"\n")
    print(f"A total of {len(rare_words)} appeared <= {min_count} times")

    with open(rare_word_file, 'w') as f:
        f.writelines(rare_words)
    
def add_context_list_to_manifest(subset: str, min_count: int):
    rare_words_file = f"data/context_biasing/{subset}_rare_words_{min_count}.txt"
    manifest_dir = f"data/fbank/libriheavy_cuts_{subset}.jsonl.gz"
    
    target_manifest_dir = manifest_dir.replace(".jsonl.gz", f"_with_context_list_min_count_{min_count}.jsonl.gz")
    if os.path.exists(target_manifest_dir):
        print(f"Target file exits at {target_manifest_dir}!")
        return
    
    print(f"Reading rare words from {rare_words_file}")
    with open(rare_words_file, "r") as f:
        rare_words = f.read()
    rare_words = rare_words.split("\n")
    rare_words = set(rare_words)
    print(f"A total of {len(rare_words)} rare words!")
    
    cuts = load_manifest_lazy(manifest_dir)
    print(f"Loaded manifest from {manifest_dir}")
    
    def _add_context(c: Cut):
        splits = remove_non_alphabetic(c.supervisions[0].texts[0], strict=False).upper().split()
        found = []
        for w in splits:
            if w in rare_words:
                found.append(w)
        c.supervisions[0].context_list = " ".join(found)
        return c

    cuts = cuts.map(_add_context)
        
    cuts.to_file(target_manifest_dir)
    print(f"Saved manifest with context list to {target_manifest_dir}")


def check(subset: str, min_count: int):
    #manifest_dir = f"data/fbank/libriheavy_cuts_{subset}_with_context_list_min_count_{min_count}.jsonl.gz"
    print("Calculating the stats over the manifest")
    manifest_dir = f"data/fbank/libriheavy_cuts_{subset}_with_context_list_min_count_{min_count}.jsonl.gz"
    cuts = load_manifest_lazy(manifest_dir)
    total_cuts = len(cuts)
    has_context_list = [c.supervisions[0].context_list != "" for c in cuts]
    context_list_len = [len(c.supervisions[0].context_list.split()) for c in cuts]
    print(f"{sum(has_context_list)}/{total_cuts} cuts have context list! ")
    print(f"Average length of non-empty context list is {sum(context_list_len)/sum(has_context_list)}")
    
if __name__=="__main__":
    #test_set = "test-clean"
    #get_facebook_biasing_list(test_set)
    subset = "medium"
    min_count = 10
    #get_rare_words(subset, min_count)
    #add_context_list_to_manifest(subset=subset, min_count=min_count)
    check(subset=subset, min_count=min_count)