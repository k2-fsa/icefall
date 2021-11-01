
dl_dir='/home/storage07/zhangjunbo/data/'
output_dir=/ceph-hw/ly/data/gigaspeech_nb/

mkdir -p $output_dir/manifests

stage=2
stop_stage=2
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Implement and verify gigaspeech downloading later"
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # subset could be: ["XS", "S", "M", "L", "XL", "DEV" "TEST"]
  # Currently only XS DEV TEST are verified
  # Others SHOULD also work
  subsets="XS DEV TEST"
  for subset in $subsets; do
    lhotse prepare gigaspeech \
      -j 60 \
      --subset=$subset \
      $dl_dir/GigaSpeech $output_dir/manifests
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 3: Compute fbank for gigaspeech"
  mkdir -p $output_dir/fbank
      ./local/compute_fbank_gigaspeech.py
fi
