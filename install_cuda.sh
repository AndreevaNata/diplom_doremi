
apt-get purge nvidia*
apt remove nvidia-*
rm /etc/apt/sources.list.d/cuda*
apt-get autoremove && apt-get autoclean
rm -rf /usr/local/cuda*

# system update
apt-get update
apt-get upgrade

# install other import packages
apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
add-apt-repository ppa:graphics-drivers/ppa -y
apt update

# install nvidia driver with dependencies
apt install libnvidia-common-515
apt install libnvidia-gl-515
apt install nvidia-driver-515

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt-get update

 # installing CUDA-11.7
apt install cuda-11-7 

# setup your paths
echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
ldconfig

# install cuDNN v11.7
# First register here: https://developer.nvidia.com/developer-program/signup

CUDNN_TAR_FILE="cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
wget https://developer.nvidia.com/compute/cudnn/secure/8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
cp -P cuda/include/cudnn.h /usr/local/cuda-11.7/include
cp -P cuda/lib/libcudnn* /usr/local/cuda-11.7/lib64/
chmod a+r /usr/local/cuda-11.7/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V