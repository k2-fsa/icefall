#!/usr/bin/env python3

import gzip
import json
import os
import shutil
import re
from pathlib import Path
from collections import defaultdict

import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
    "--teacher-model-id",
    type=str,
    default="zipformer_l_56",
    help="could be one of:  zipformer_l_56",
    )

    parser.add_argument(
    "--output-path",
    type=str,
    default="mvq_kd_zipformer/exp",
    help="combined jsonl file output path",
    )

    parser.add_argument(
        "--distillation-layer",
        type=str,
        default="1,7",
        help="Distillation layer index of student model",
    )

    parser.add_argument(
        "--num-codebooks",
        type=str,
        default="16,16",
        help="Used to construct distillation loss",
    )

    parser.add_argument(
        "--use-mul-tea",
        type=bool,
        default=False,
        help="If True, use multi-teacher codebook index.",
    )

    return parser

def process_files(input_paths, teacher_ids, output_path):
    # 初始化聚合数据结构
    merged_data = defaultdict(dict)
    
    # 遍历每个教师文件
    for teacher_id, file_path in zip(teacher_ids, input_paths):
        print(f"start process {teacher_id}")
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                batch_idx = data['batch_idx']
                
                # 生成新字段名 
                new_entry = {
                    f"{key}_{teacher_id}": value 
                    for key, value in data.items()
                    if key != 'batch_idx'
                }
                
                # 合并到主数据 
                merged_data[batch_idx].update(new_entry)
                merged_data[batch_idx]['batch_idx'] = batch_idx
    
    # 写入最终文件 
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for batch in sorted(merged_data.values(), key=lambda x: x['batch_idx']):
            f.write(json.dumps(batch) + '\n')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.output_path = Path(args.output_path)
    input_jsonl_path = []
    teacher_id_list = ["zipformer_s_55", "zipformer_m_55", "zipformer_l_56"]
    for i, each_teacher in enumerate(teacher_id_list):
        input_jsonl_path.append(f"{args.output_path}/combined_1_7/{each_teacher}_loss_logs.jsonl.gz")

    output_path = f"{args.output_path}/combined_1_7/merged_logs_{len(teacher_id_list)}teachers.jsonl.gz"
    process_files(input_jsonl_path, teacher_id_list, output_path)
    print("Done ! ")