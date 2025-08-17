#!/usr/bin/env python3
# extract list of sph from a cut jsonl
# python3 extract_list_of_sph.py dev_cuts_swbd.jsonl > data/fbank/dev_swbd_sph.list


import sys, json

inputfile = sys.argv[1]
json_str = [line.rstrip("\n") for line in open(inputfile)]
num_json = len(json_str)

for i in range(num_json):
    if json_str[i] != "":
        cur_json = json.loads(json_str[i])
        for keys in cur_json:
            cur_rec = cur_json["recording"]
            cur_sources = cur_rec["sources"]
            for s in cur_sources:
                cur_sph = s["source"]
                print(cur_sph)
