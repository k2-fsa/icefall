#!/usr/bin/env python3

# 


import sys, json ;
import ntpath;

list_of_sph = sys.argv[1];
jsonfile = sys.argv[2];
out_partition_json = sys.argv[3];


list_of_sph=[line.rstrip('\n') for line in open(list_of_sph)]

sph_basename_list=[]

for f in list_of_sph:
    bsname=ntpath.basename(f)
    #print(bsname)
    sph_basename_list.append(ntpath.basename(f))


json_str=[line.rstrip('\n') for line in open(jsonfile)]
num_json = len(json_str)

#cutid2sph=dict()
out_partition=open(out_partition_json,'w',encoding='utf-8')

for i in range(num_json):
    if json_str[i] != '':
        #print(json_str[i])
        cur_json = json.loads(json_str[i])
        #print(cur_json)
        cur_cutid= cur_json['id']
        cur_rec = cur_json['recording']
        cur_sources = cur_rec['sources']        
        #print(cur_cutid)
        #print(cur_rec)
        #print(cur_sources)
        for s in cur_sources:
            cur_sph = s['source']
            cur_sph_basename=ntpath.basename(cur_sph)
        #print(cur_sph)
        #print(cur_sph_basename)
        if cur_sph_basename in sph_basename_list :
            out_json_line = json_str[i]
            out_partition.write(out_json_line)
            out_partition.write("\n")
        #for keys in cur_json:
            #cur_cutid= cur_json['id']
            #cur_rec = cur_json['recording_id']
            #print(cur_cutid)
            
    
"""
        for keys in cur_json:
            #print(keys)
            cur_cutid= cur_json['id']
            cur_rec = cur_json['recording_id']
            print(cur_rec)
            if cur_sph_basename in sph_basename_list :
                out_json_line = json_str[i]
                out_partition.write(out_json_line)
                out_partition.write("\n")
"""
        
        




