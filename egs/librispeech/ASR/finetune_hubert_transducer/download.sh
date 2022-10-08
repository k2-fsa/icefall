#!/usr/bin/env bash

# Usage:
# Navigate to /path/to/egs/lirbispeech/ASR, then call
# ./finetune_hubert_transducer/download.sh model_type
# replace model_type with the model type you want
# Available types are [base, large, xlarge]
# For example, if you want to download Hubert large, run:
# ./finetune_hubert_transducer/download.sh large

model_type=$1

download_dir="pretrained_models"
cd finetune_hubert_transducer
mkdir -p ${download_dir}

# download pretrained models
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

# download dict file
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -P ${download_dir}


# url_dict = {
#         "base": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
#         "large": "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt",
#         "xlarge": "https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt"
#     }
