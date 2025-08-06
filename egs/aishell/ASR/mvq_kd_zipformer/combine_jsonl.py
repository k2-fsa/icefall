#!/usr/bin/env python3

import gzip
import json
import os
import shutil
import re
from pathlib import Path
from collections import defaultdict
from icefall.utils import str2bool

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
        type=str2bool,
        default=False,
        help="If True, use multi-teacher codebook index.",
    )

    return parser

def merge_codebooks(input_paths, output_path):
    # 按cut_id聚合所有codebook数据
    cut_store = defaultdict(dict)

    # 创建输出目录（新增关键修复）
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有输入文件
    for input_file in input_paths:
        # 从路径提取层级和码本数: layer1_cb16 → ('1', '16')
        layer = re.search(r'layer(\d+)_cb(\d+)', input_file).groups()
        codebook_key = f"codebook_indexes_layer{layer[0]}_cb{layer[1]}"
        
        # 读取当前文件的所有codebook数据
        with gzip.open(input_file, 'rt') as f:
            for line in f:
                data = json.loads(line)
                cut_id = data['id']
                # 提取当前文件的codebook数据
                if 'custom' in data and 'codebook_indexes' in data['custom']:
                    codebook_data = data['custom']['codebook_indexes']
                    # 存入聚合存储（确保同一cut不同层数据合并）
                    cut_store[cut_id][codebook_key] = codebook_data
    
    # 生成最终合并文件
    with gzip.open(output_path, 'wt') as f_out:
        # 以第一个文件为基础结构（确保非custom字段完整）
        with gzip.open(input_paths[0], 'rt') as f_base:
            for line in f_base:
                base_data = json.loads(line)
                cut_id = base_data['id']
                
                # 合并当前cut的所有codebook数据
                if cut_id in cut_store:
                    if 'custom' not in base_data:
                        base_data['custom'] = {}
                    base_data['custom'].update(cut_store[cut_id])
                    
                    # 写入合并后的完整数据
                    f_out.write(json.dumps(base_data, ensure_ascii=False) + '\n')

def copy_symlinks(src_path, dst_path):
    # 遍历输入路径下的所有内容
    for root, dirs, files in os.walk(src_path):
        # 计算目标路径的对应目录结构
        relative_path = os.path.relpath(root, src_path)
        dst_dir = os.path.join(dst_path, relative_path)
        
        # 创建目标目录结构（若不存在）
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        
        # 处理文件中的软链接
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            
            # 仅处理软链接文件
            if os.path.islink(src_file):
                # 获取原链接指向的路径
                link_target = os.readlink(src_file)
                
                # 计算相对路径（确保目标路径有效性）
                if not os.path.isabs(link_target):  # 若原链接是相对路径
                    # 获取原链接相对于输入路径的绝对路径
                    abs_link = os.path.abspath(os.path.join(root, link_target))
                    # 转换为相对于目标目录的相对路径
                    rel_link = os.path.relpath(abs_link, os.path.dirname(dst_file))
                else:  # 若原链接是绝对路径
                    rel_link = link_target
                
                # 创建新的软链接
                if os.path.lexists(dst_file):  # 若目标路径已存在则删除
                    os.remove(dst_file)
                os.symlink(rel_link, dst_file)

def merge_teachers(teacher_id_list, input_jsonl_paths, output_path):
    # 使用字典合并相同ID的记录
    merged_data = defaultdict(dict)

    for teacher_id, jsonl_path in zip(teacher_id_list, input_jsonl_paths):
        print(f"start combine {teacher_id}   .  . . ")
        with gzip.open(jsonl_path, 'rt') as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line.strip())
                entry_id = data['id']
                
                # 初始化条目结构
                if not merged_data[entry_id]:
                    # 复制基础字段：排除需要合并的custom字段
                    base_fields = {k: v for k, v in data.items() if k != 'custom'}
                    merged_data[entry_id] = base_fields
                    merged_data[entry_id]['custom'] = {}

                # 处理custom字段
                custom = data.get('custom', {})
                
                # 添加教师后缀到指定字段
                for key in ['codebook_indexes_layer1_cb16', 
                          'codebook_indexes_layer7_cb16']:
                    if key in custom:
                        new_key = f"{key}_{teacher_id}"
                        merged_data[entry_id]['custom'][new_key] = custom[key]
                        
                # 保留原始custom的其他字段
                for k, v in custom.items():
                    if k not in ['codebook_indexes_layer1_cb16',
                               'codebook_indexes_layer7_cb16']:
                        if k not in merged_data[entry_id]['custom']:
                            merged_data[entry_id]['custom'][k] = v

    # 写入合并后的文件
    with gzip.open(output_path, 'wt') as f_out:
        for entry in merged_data.values():
            # 保持JSON序列化的一致性
            formatted_entry = json.dumps(entry, ensure_ascii=False, indent=None)
            f_out.write(formatted_entry + '\n')

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.output_path = Path(args.output_path)
    # print("output-path : ", args.output_path)
    # 配置示例,包含单层codebook index的jsonl.gz文件
    input_files = []
    layer = []
    for num_codebooks, distillation_layer in zip(tuple(map(int, args.num_codebooks.split(","))), tuple(map(int, args.distillation_layer.split(",")))):
        # input_files.append(f"{args.output_path}/{distillation_layer}_cb{num_codebooks}/aishell_cuts_train-train.jsonl.gz")
        input_files.append(f"{args.output_path}/vq/{args.teacher_model_id}_layer{distillation_layer}_cb{num_codebooks}/aishell_cuts_train-train.jsonl.gz")
        layer.append(distillation_layer)

    output_path = f"{args.output_path}/combined_{layer[0]}_{layer[1]}/combined_output_{layer[0]}_{layer[1]}_{args.teacher_model_id}.jsonl.gz"
    if not args.use_mul_tea:
        # 执行处理
        merge_codebooks(input_files, output_path)
        print(f"合并完成 -> {output_path}")
        src_path = os.path.dirname(input_files[0])
        dst_path = os.path.dirname(output_path)
        copy_symlinks(src_path, dst_path)
        print(f"复制软链接完成 -> {dst_path}")
    else:
        print("start combine multi-teacher codebook index . . .")
        input_jsonl_path = []
        teacher_id_list = ["zipformer_s_55", "zipformer_m_55", "zipformer_l_56"]
        for i, each_teacher in enumerate(teacher_id_list):
            input_jsonl_path.append(f"{args.output_path}/combined_{layer[0]}_{layer[1]}/combined_output_{layer[0]}_{layer[1]}_{each_teacher}.jsonl.gz")
        # print(input_jsonl_path)
        output_path = f"{args.output_path}/combined_{layer[0]}_{layer[1]}/combined_output_{layer[0]}_{layer[1]}_{len(teacher_id_list)}teachers.jsonl.gz"
        merge_teachers(teacher_id_list, input_jsonl_path, output_path)
        print("Done ! ")
