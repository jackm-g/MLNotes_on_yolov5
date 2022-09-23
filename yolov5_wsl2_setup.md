# Setting up Yolov5 on WSL2 for GPU Usage
Updated 9/23/2022

## Sytem Specs
GPU: NVIDIA 3080

NVDIA Driver: 516.94

CUDA: 11.2

PyTorch Libariries: torch 1.9.0+cu111, torchvision 0.10.0+cu111

## Update NVIDIA Drivers
Update your NVIDIA drivers to the latest available version that will work with CUDA 11.2, so any driver with 11.X compatibility.
## Install WSL2
Follow the steps listed here https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl, then here https://docs.nvidia.com/cuda/wsl-user-guide/index.html

    - Windows Version 21H2 or higher required for CUDA compatibility.
        - Run the `winver` application in Windows to see current version.
    - Test that CUDA is working with `nvidia-smi` command. Note that CUDA Version lists compatiblity, not the actual installed version.

```
$ nvidia-smi
Fri Sep 23 11:11:54 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 516.94       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:06:00.0  On |                  N/A |
|  0%   48C    P8    35W / 320W |   2072MiB / 10240MiB |     26%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
```

## Get Yolov5 Repository
Clone the yolov5 repository to your WSL instance and follow the instructions in the README to install dependencies.
`$ git clone https://github.com/ultralytics/yolov5.git`

OPTIONAL: I used virtualenv to prepare a fresh pip environment for all the dependencies.
```
$ python3 -m venv /path/to/new/virtual/environment
$ source activate /path/to/new/virtual/environment/bin/activate
```

## Find working PyTorch
My initial pytorch version installed through `requirements.txt` was not compatible with CUDA 11.2.

I found a working version from this github issue comment: https://github.com/pytorch/pytorch/issues/50032#issuecomment-885423131

Using that, running the below installed a working version
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Run Training