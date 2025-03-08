import torch
import torchaudio

print("PyTorch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("GPU name:", torch.cuda.get_device_name(0))
