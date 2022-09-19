# icefall dockerfile

2 sets of configuration are provided - (a) Ubuntu18.04-pytorch1.12.1-cuda11.3-cudnn8, and (b) Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8.

If your NVIDIA driver supports CUDA Version: 11.3, please go for case (a) Ubuntu18.04-pytorch1.12.1-cuda11.3-cudnn8. Otherwise, the older PyTorch images are not updated with the [apt-key rotation by NVIDIA](https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key), so you have to go for case (b) Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8.

You can check the highest CUDA version within your NVIDIA driver's support with the `nvidia-smi` command below. In this example, the highest CUDA version is 10.0, i.e. case (b) Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8.

```bash
$ nvidia-smi
Wed Nov 21 19:41:32 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.72       Driver Version: 410.72       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 106...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   53C    P0    26W /  N/A |    379MiB /  6078MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1324      G   /usr/lib/xorg/Xorg                           225MiB |
|    0      2844      G   compiz                                       146MiB |
|    0     15550      G   /usr/lib/firefox/firefox                       1MiB |
|    0     19992      G   /usr/lib/firefox/firefox                       1MiB |
|    0     23605      G   /usr/lib/firefox/firefox                       1MiB |

```

## Building images locally
If your environment requires a proxy to access the Internet, remember to add those information into the Dockerfile directly. 
For most cases, you can uncomment these lines in the Dockerfile and add in your proxy details. 

```dockerfile
ENV http_proxy http://aaa.bb.cc.net:8080
ENV https_proxy http://aaa.bb.cc.net:8080
```

Then, proceed with these commands. 

### If you are case (a), i.e. your NVIDIA driver supports CUDA version >= 11.3:

```bash
cd docker/Ubuntu18.04-pytorch1.12.1-cuda11.3-cudnn8
docker build -t icefall/pytorch1.12.1 .
```

### If you are case (b), i.e. your NVIDIA driver can only support CUDA versions <11.3:
```bash
cd docker/Ubuntu18.04-pytorch1.7.1-cuda11.0-cudnn8
docker build -t icefall/pytorch1.7.1 .
```

## Running your built local image
Sample usage of the GPU based images. These commands are written with case (a) in mind, so please make the necessary changes to your image name if you are case (b).
Note: use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) to run the GPU images.

```bash
docker run -it --runtime=nvidia --shm-size=2gb --name=icefall --gpus all icefall/pytorch1.12.1
```

### Tips:
1. Since your data and models most probably won't be in the docker, you must use the -v flag to access the host machine. Do this by specifying `-v {/path/in/docker}:{/path/in/host/machine}`. 

2. Also, if your environment requires a proxy, this would be a good time to add it in too: `-e http_proxy=http://aaa.bb.cc.net:8080 -e https_proxy=http://aaa.bb.cc.net:8080`.

Overall, your docker run command should look like this. 

```bash
docker run -it --runtime=nvidia --shm-size=2gb --name=icefall --gpus all -v {/path/in/docker}:{/path/in/host/machine} -e http_proxy=http://aaa.bb.cc.net:8080 -e https_proxy=http://aaa.bb.cc.net:8080 icefall/pytorch1.12.1
```

You can explore more docker run options [here](https://docs.docker.com/engine/reference/commandline/run/) to suit your environment.

### Linking to icefall in your host machine

If you already have icefall downloaded onto your host machine, you can use that repository instead so that changes in your code are visible inside and outside of the container. 

Note: Remember to set the -v flag above during the first run of the container, as that is the only way for your container to access your host machine. 
Warning: Check that the icefall in your host machine is visible from within your container before proceeding to the commands below.

Use these commands once you are inside the container.

```bash
rm -r /workspace/icefall
ln -s {/path/in/docker/to/icefall} /workspace/icefall
```

## Starting another session in the same running container.
```bash
docker exec -it icefall /bin/bash
```

## Restarting a killed container that has been run before. 
```bash
docker start -ai icefall
```

## Sample usage of the CPU based images:
```bash
docker run -it icefall /bin/bash
``` 
