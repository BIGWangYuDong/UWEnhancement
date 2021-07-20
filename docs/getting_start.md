## Prerequisites

Our development environment is:

- Linux (Ubuntu 16.04 or 18.04)
- Python 3.6
- CUDA 10.1

**NOTE:** 

1. The main code should be able to support in Windows and macOS. (We did not experiment, but it should be possible to run if the environment is successfully installed.)

2. CUDA version should be greater than 9.0 (CUDA >=9.0) so that can install [apex](https://github.com/NVIDIA/apex). 

## Installation

We recommend using anaconda to create a new environment.

0. Clone the repo
      ```shell
   git clone https://github.com/BIGWangYuDong/UWEnhancement.git UW
      ```

1. Create a conda virtual environment and activate it.

     ```shell
   conda create -n UW python=3.6 -y
   conda activate UW
     ```

2.  Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

   ```shell
   conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
   ```

3. Install build requirements and install addict.

   ```shell
   pip install -r requirements.txt
   pip install addict
   ```

4. We support fp16, if want to use fp16 during training or testing, you can install apex.

   ```shell
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   # If the installation fails, you can try :
   python setup.py install
   ```
