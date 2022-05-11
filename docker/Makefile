build_docker:
	docker build -t icefall/pytorch1.7.1:latest -f ./Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8/Dockerfile ./

run_docker:
	docker run -it --rm --runtime=nvidia \
		--gpus all \
		-v /data1:/data1 \
		-v /data1/merge_all_short/raw/fr_token_list/bpe_unigram5000/bpe.pyonmttok.vocab:/data/vocab \
		-v /data1/merge_all_short/raw/fr_token_list/bpe_unigram5000/bpe.pyonmttok:/data/bpe.pyonmttok \
		-v /nas-labs/ASR/valentin_work/icefall:/workspace/icefall \
		--name val_icefall_3 icefall/pytorch1.7.1:latest bash