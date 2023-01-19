# usage:
# python3 utils/decoded_result_to_text_files.py input_file output_directory

import os
import sys
import re

input_file = sys.argv[1]
output_directory = sys.argv[2]

if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

with open(input_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "_sp" in line:
            continue
        # use regular expression to extract the filename and hyp list
        match = re.search(r'(.*):\s*hyp=\[(.*)\]', line)
        if match:
            filename = match.group(1)
            hyp = match.group(2)
            hyp = hyp.replace("'", "").replace("[", "").replace("]", "").split(',')
            hyp_sentence = ' '.join(hyp)

            filename = "-".join(filename.split('-')[:-1])
            output_file = os.path.join(output_directory, f'{filename}.txt')

            with open(output_file, 'w') as f:
                f.write(hyp_sentence)