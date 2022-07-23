#!/usr/bin/env python3

# python3 extract_list_of_sph.py dev_cuts_swbd.jsonl > data/fbank/dev_swbd_sph.list


import sys, json ;
inputfile = sys.argv[1]
json_str=[line.rstrip('\n') for line in open(inputfile)]
num_json = len(json_str)

#print(num_json)
#with open(inputfile, 'r',encoding='utf-8') as Jsonfile:
#    print("Converting JSON encoded data into Python dictionary")
#    json_dict = json.load(Jsonfile)
#    for k,v in json_dict:
#        print(k,v)



for i in range(num_json):
    if json_str[i] != '':
        #print(json_str[i])
        cur_json = json.loads(json_str[i])
        # print(cur_json)
        for keys in cur_json:
            #print(keys)
            cur_rec = cur_json['recording']
            cur_sources = cur_rec['sources']
            #print(cur_sources)
            for s in cur_sources:
                cur_sph = s['source']                
                print(cur_sph)
            #cur_sph = cur_sources[2]
            #print(cur_sph)

    

#print(json.load(sys.stdin)['source'])
