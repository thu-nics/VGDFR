# VGDFR: Diffuison-based Video Generation with Dynamic Frame Rate
This is the official implementation of the paper "VGDFR: Diffuison-based Video Generation with Dynamic Frame Rate".

## Installation

```bash
# 1. Create conda environment
conda create -n VGDFR python==3.10.9

# 2. Activate the environment
conda activate VGDFR

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install flash attention for acceleration
conda install cuda-nvcc
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4

# 6. Install xDiT for parallel inference (It is recommended to use torch 2.4.0 and flash-attn 2.6.3)
python -m pip install xfuser==0.4.3

```
