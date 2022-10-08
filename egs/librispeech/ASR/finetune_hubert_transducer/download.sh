#!/usr/bin/env bash

model_type=$1

download_dir="pretrained_models"
cd finetune_hubert_transducer
mkdir -p ${download_dir}

if [ "${model_type}" = "base" ]; then
    if [ -f "${download_dir}/hubert_base_ls960.pt" ]; then
        echo "File already exist, skip downloading"
    else
        wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -P ${download_dir}
        echo "Finished downloading Hubert-${model_type}"
    fi
elif [ "${model_type}" = "large" ]; then
    if [ -f "${download_dir}/hubert_large_ll60k.pt" ]; then
        echo "File already exist, skip downloading"
    else
        wget https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt -P ${download_dir}
        echo "Finished downloading Hubert-${model_type}"
    fi
elif [ "${model_type}" = "xlarge" ]; then
    if [ -f "${download_dir}/hubert_xtralarge_ll60k.pt" ]; then
        echo "File already exist, skip downloading"
    else
        wget https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt -P ${download_dir}
        echo "Finished downloading Hubert-${model_type}"
    fi
else
    echo "Unknown Hubert type: ${model_type}"
fi

# url_dict = {
#         "base": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
#         "large": "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt",
#         "xlarge": "https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt"
#     }
