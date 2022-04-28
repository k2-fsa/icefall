stage=4

# Parameters about model.
exp_dir=./vq_pruned_transducer_stateless2/exp/
model_id=hubert_xtralarge_ll60k_finetune_ls960
hubert_model_dir=${exp_dir}/hubert_models
hubert_model=${hubert_model_dir}/${model_id}.pt

# Parameters about quantizer.
num_utts=1000
mem_layer=36
bytes_per_frame=8
enable_refine=True

if [ $stage -eq -1 ]; then
  # Preparation state.

  # Install fairseq according to:
  # https://github.com/pytorch/fairseq
  # when testing this code:
  # commit 806855bf660ea748ed7ffb42fe8dcc881ca3aca0 is used

  echo "Download hubert model."
  mkdir -p ${hubert_model_dir}
  # For more models refer to: https://github.com/pytorch/fairseq/tree/main/examples/hubert
  wget -c https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k_finetune_ls960.pt -P ${hubert_model_dir}
  wget -c wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -P ${hubert_model_dir}
fi


if [ $stage -eq 0 ]; then
  # This stage is not directly used by codebook extraction.
  # It is an method to "prove" that the downloaed hubert model
  # is inferenced in an correct way if WERs looks like normal.
  # Expect WERs:
  # [test-clean-ctc_greedy_search] %WER 2.04% [1075 / 52576, 92 ins, 104 del, 879 sub ]
  # [test-other-ctc_greedy_search] %WER 3.71% [1942 / 52343, 152 ins, 126 del, 1664 sub ]
  export CUDA_VISIBLE_DEVICES=7
  ./vq_pruned_transducer_stateless2/hubert_decode.py \
    --max-duration 10
fi

if [ $stage -eq 1 ]; then
  ./vq_pruned_transducer_stateless2/hubert_memory_embeddings.py \
    --max-duration 10
fi

if [ $stage -eq 2 ]; then
  ./vq_pruned_transducer_stateless2/quantizer_train.py
fi

# CAUTITHON: set quantizer_id MANUALLY when a new quantizer is used.
# as it is generated randomly.
# quantizer_id="ba401508"; max_duration=40;
quantizer_id="3d451334"; max_duration=40;

# Train with clean-100
train_subsets="clean-100"
# Or if full-libri speech is needed:
# train_subsets="clean-100 clean-360 other-500"
# In stage 4, each split part needs a gpu to extract codebook indexes.
# So use a larger num_jobs if more GPUs are available.
num_jobs=2
manifests_dir=vq_pruned_transducer_stateless2/exp/manifests/
if [ $stage -eq 3 ]; then
  for subset in ${train_subsets}; do
    echo $subset
    split_dir=$manifests_dir/split${num_jobs}/$subset/
    mkdir -p  $split_dir
    lhotse split $num_jobs data/fbank/cuts_train-$subset.json.gz  $split_dir
  done
fi

if [ $stage -eq 4 ]; then
  refine_iter=5

  extract_codebook_index(){
    # When I testing this code, gpu 6 and 7 are available,
    # So the CUDA_VISIBLE_DEVICES is (1 + 5) for job 0
    # and (2 + 5) for job 1
    # Note: order of split manfiests is 1-based, while gpu is 0-based.
    export CUDA_VISIBLE_DEVICES=`(expr $1 + 5)`
    ./vq_pruned_transducer_stateless2/hubert_code_indices.py \
      --num-splits $num_jobs \
      --subset=$2 \
      --manifest-idx $1 \
      --ori-manifest-dir=$manifests_dir/split${num_jobs}/$subset/ \
      --max-duration=$max_duration \
      --quantizer-id=${quantizer_id}
  }
  # With two pieces of NVIDIA A100 gpus, around three hours needed to process 300 hours training data,
  # i.e. clean-100 with speed purteb 0.9 and 1.1.
  for subset in ${train_subsets}; do
    for manifest_idx in `seq 1 $num_jobs`; do
      extract_codebook_index $manifest_idx $subset &
    done
    wait
  done
  wait
fi
if [ $stage -eq 5 ]; then
  for subset in ${train_subset}; do
    cdidx_manifests_dir=`pwd`/data/$model_id-${mem_layer}layer-${quantizer_id}-bytes_per_frame-${bytes_per_frame}
    combined_list=`find $cdidx_manifests_dir/splits$num_jobs/ -name cuts_train-${sbuset}*`
    echo $combined_list
    lhotse combine $combined_list $cdidx_manifests_dir/cuts_train-${subset}.json.gz
  done
fi
