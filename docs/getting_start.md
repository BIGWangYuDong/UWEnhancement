conda create -n 环境名 python=3.7 -y

按照mmdetection 安装

conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

pip install mmcv-full==1.1.1

pip install -r requirements/build.txt
pip install -v -e .
