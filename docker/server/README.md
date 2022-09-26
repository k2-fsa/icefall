docker build -t backend_icefall:valentin -f docker/server/Dockerfile ./
docker build -t backend_icefall_us:valentin -f docker/server/Dockerfile-us ./
docker build -t backend_icefall_azure:valentin -f docker/server/Dockerfile-azure ./

docker run -it --rm -p 6008:6008 backend_icefall:valentin
docker run -it --rm -p 6008:6008 backend_icefall_us:valentin

docker tag backend_icefall:valentin backend_icefall:prod

docker push 

docker run -it --rm --cap-add SYS_PTRACE -p 6008:6008 -v /nas-labs/ASR/valentin_work/icefall/egs/librispeech/ASR/transducer_emformer_pyonmttok/server_azure_fusion.py:/workspace/egs/librispeech/ASR/transducer_emformer_pyonmttok/server.py -v /nas-labs/ASR/valentin_work/icefall/credentials.json:/cred/credentials.json --env-file /nas-labs/ASR/valentin_work/icefall/docker/server/env-variable.list backend_icefall:valentin

env-variable.list : GOOGLE_DOCUMENT_ID and AZURE_SPEECH_KEY