import argparse 
import jiwer
import os 
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dec-file",
        type=str,
        help="file with decoded text"
    )

    return parser


def contains_chinese(text):
    """
    Check if the given text contains any Chinese characters.

    Args:
        text (str): The input string.

    Returns:
        bool: True if the string contains at least one Chinese character, False otherwise.
    """
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_char_pattern.search(text))

def cer_(file):
    hyp = []
    ref = []
    cer_results = 0
    ref_lens = 0
    with open(file, 'r', encoding='utf-8') as dec:
        
        for line in dec:
            id, target = line.split('\t')
            id = id[0:-1]
            target, txt = target.split("=")
            
            if target == 'ref':
                words = txt.strip().strip('[]').split(', ')
                word_list = [word.strip("'") for word in words]
                # if contains_chinese(" ".join(word_list)):
                #     word_list = [" ".join(re.findall(r".",word.strip("'"))) for word in words]
                ref.append("".join(word_list))
            elif target == 'hyp':
                words = txt.strip().strip('[]').split(', ')
               
                word_list = [word.strip("'") for word in words]
                # if contains_chinese(" ".join(word_list)):
                    
                #     word_list = ["".join(re.findall(r".",word.strip("'"))) for word in words]
                hyp.append("".join(word_list))
        for h, r in zip(hyp, ref):
            if r:
                cer_results += (jiwer.cer(r, h)*len(r))
                
                ref_lens += len(r)
    #print(os.path.basename(file))
    print(cer_results / ref_lens)

        


def main():
    parse = get_args()
    args = parse.parse_args() 
    cer_(args.dec_file)
    
if __name__ == "__main__":
    main()