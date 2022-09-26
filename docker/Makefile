build_docker:
	docker build -t icefall/pytorch1.7.1-188:latest -f ./Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8-188/Dockerfile ./

run_docker:
	docker run -it --rm --runtime=nvidia \
		--gpus all \
		-v /data1:/data1 \
		-v /data1/merge_all_short/raw/fr_token_list/bpe_unigram5000/bpe.pyonmttok.vocab_blk:/data/vocab \
		-v /data1/merge_all_short/raw/fr_token_list/bpe_unigram5000/bpe.pyonmttok:/data/bpe.pyonmttok \
		-v /nas-labs/LM:/nas-labs/LM \
		-v /nas-labs/ASR/valentin_work/icefall:/workspace/icefall \
		-v /nas-labs/ASR/valentin_work/icefall:/nas-labs/ASR/valentin_work/icefall \
		--net host \
		--shm-size 8G \
		--name val_icefall_2 icefall/pytorch1.7.1-188:latest bash

run_docker_189:
	docker run -it --rm --runtime=nvidia \
		--gpus all \
		-v /data1:/data1 \
		-v /nas-labs:/nas-labs \
		-v /nas-labs/ASR/valentin_work/icefall:/workspace/icefall \
		-v /nas-labs/ASR/valentin_work/icefall:/nas-labs/ASR/valentin_work/icefall \
		--net host \
		--shm-size 8G \
		--name val_icefall icefall/pytorch1.7.1-188:latest bash