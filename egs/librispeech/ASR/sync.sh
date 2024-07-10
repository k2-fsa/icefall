project=icefall-asr-librispeech-zipformer-2023-11-04
run=4V10032G_lstm1_decoderdropout0.2_bpe500
recipe=zipformer_lstm

wandb sync ${recipe}/exp_dropout0.2/tensorboard/ --sync-tensorboard  -p $project  --id $run

while true
do
  wandb sync ${recipe}/exp_dropout0.2/tensorboard/ --sync-tensorboard  -p $project  --id $run  --append
  sleep 60
done
