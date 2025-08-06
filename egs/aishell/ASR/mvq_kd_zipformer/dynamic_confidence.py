import gzip
import json
import torch
# 写入数据到jsonl.gz文件（标准库实现）
def write_logs_to_gz(filename, logs):
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        for log in logs:
            # 转换Tensor为Python原生类型
            log_dict = {
                "batch_idx": log["batch_idx"],
                "feature_shape": list(log["feature_shape"]),  # 元组转列表
                "simple_loss": log["simple_loss"].item() if isinstance(log["simple_loss"], torch.Tensor) else log["simple_loss"],
                "pruned_loss": log["pruned_loss"].item() if isinstance(log["pruned_loss"], torch.Tensor) else log["pruned_loss"],
                "loss_per_batch": log["loss_per_batch"].item() if isinstance(log["loss_per_batch"], torch.Tensor) else log["loss_per_batch"],
                "loss_per_batch_per_frame": log["loss_per_batch_per_frame"].item() if isinstance(log["loss_per_batch_per_frame"], torch.Tensor) else log["loss_per_batch_per_frame"],
                "distillation_layers_sharpness": log[ "distillation_layers_sharpness"],
            }
            f.write(json.dumps(log_dict) + '\n')

# 读取jsonl.gz文件（标准库实现）
def read_logs_from_gz(filename):
    logs = []
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            log = json.loads(line.strip())
            # 可选：将shape转回元组
            # log["feature_shape"] = tuple(log["feature_shape"])
            logs.append(log)
    return logs


def write_dkloss_to_gz(filename: str, formatted_total_losses_to_gz: list) -> None:
    # 使用追加模式写入文件，保留历史数据
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        json_line = json.dumps(formatted_total_losses_to_gz)
        f.write(json_line)

def read_dkloss_from_gz(filename: str) -> list:
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    return data