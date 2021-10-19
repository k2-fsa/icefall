# icefall dockerfile

We provide a dockerfile for some users, the configuration of dockerfile is : Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8-python3.8. You can use the dockerfile by following the steps:

## Building images locally

```bash
cd docker/Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8
docker build -t icefall/pytorch1.7.1:latest -f ./Dockerfile ./
```

## Using built images 
Sample usage of the GPU based images:
Note: use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the GPU images.

```bash
docker run -it --runtime=nvidia --name=icefall_username --gpus all icefall/pytorch1.7.1:latest
```

Sample usage of the CPU based images:

```bash
docker run -it icefall/pytorch1.7.1:latest /bin/bash
``` 