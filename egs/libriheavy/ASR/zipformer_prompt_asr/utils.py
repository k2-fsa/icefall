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

def get_rare_words():
    txt_path = f"data/lang_bpe_500/transcript_words_{subset}.txt"
    rare_word_file = f"data/context_biasing/{subset}_rare_words_{min_count}.txt"
    
    if os.path.exists(rare_word_file):
        print("File exists, do not proceed!")
        return
    with open(txt_path, "r") as file:
        words = file.read().upper().split()
        word_count = {}
        for word in words:
            word = remove_non_alphabetic(word, strict=False)
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    print(f"A total of {len(word_count)} words appeared!")
    rare_words = []
    for k in word_count:
        if word_count[k] <= min_count:
            rare_words.append(k+"\n")
    print(f"A total of {len(rare_words)} appeared <= 10 times")

    with open(rare_word_file, 'w') as f:
        f.writelines(rare_words)
    
def add_context_list_to_manifest(subset: str, min_count: int):
    rare_words_file = f"data/context_biasing/{subset}_rare_words_{min_count}.txt"
    manifest_dir = f"data/fbank/librilight_cuts_train_{subset}.jsonl.gz"
    
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
        splits = remove_non_alphabetic(c.supervisions[0].text).upper().split()
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
    manifest_dir = f"data/fbank/librilight_cuts_train_{subset}_with_context_list_min_count_{min_count}.jsonl.gz"
    cuts = load_manifest_lazy(manifest_dir)
    total_cuts = len(cuts)
    has_context_list = [c.supervisions[0].context_list != "" for c in cuts]
    print(f"{sum(has_context_list)}/{total_cuts} cuts have context list! ")
    
    
    
if __name__=="__main__":
    #test_set = "test-clean"
    #get_facebook_biasing_list(test_set)
    #get_rare_words()
    subset = "small"
    add_context_list_to_manifest(subset=subset)