
WANDB_KEY=df59308c1f07be8338a87497523163014442d605   # TODO Set YOUR KEY!
wandb login ${WANDB_KEY}
torchrun --nproc_per_node=8 train.py config.json
