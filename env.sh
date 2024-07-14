conda create -n rein -y python=3.8
conda install -n rein pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
/opt/conda/envs/rein/bin/pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
/opt/conda/envs/rein/bin/pip install -U openmim
/opt/conda/envs/rein/bin/mim install mmengine
/opt/conda/envs/rein/bin/mim install "mmcv==2.0.0"
/opt/conda/envs/rein/bin/pip install "mmsegmentation>=1.0.0"
/opt/conda/envs/rein/bin/pip install "mmdet>=3.0.0"
/opt/conda/envs/rein/bin/pip install "xformers==0.0.20"
cd /mnt/csip-090/rein
/opt/conda/envs/rein/bin/pip install -r requirements.txt
/opt/conda/envs/rein/bin/pip install future tensorboard
ln -s /mnt/csip-090/rein/work_dirs /mnt/tensorboard
