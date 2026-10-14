# Visual Speech Recognition - Grid Corpus - AV-HuBERT features

### Model Description
This model is a reworked CTC-attention architecture based on librispeech recipe.
### Input Processing
Mouth regions of interest (ROIs) are tracked and extracted at 64x64 pixels using dlib, then upsampled to 88x88 pixels to meet AV-HuBERT's input requirements. Features are extracted from layer 9 of the pretrained AV-HuBERT model (base_vox_iter5.pt) and projected to 256 dimensions.
### Architecture
The convolution/subsampling stage is omitted, preserving visual features at their native 25 fps temporal resolution. The encoder consists of 4 reworked Conformer layers and 2 reworked Transformer decoder layers.

### Task & Data Split
The model is evaluated on the unseen speaker task. Speakers S1, S2, S20, and S22 are held out exclusively for testing and are not seen during training.

### Getting Started:
Python 3.8 is required to ensure compatibility with AV-HuBERT and its dependencies, as newer Python versions may introduce issues or unsupported changes.

#### Create a conda environment before running install.sh
```
conda create -n icefall-vsr python=3.8 -y
conda activate icefall-vsr
```
