# mesh_utils

Utils for meshes.

### Setting up a conda environment that works

conda create --name mesh_utils python=3.6.7 pip
conda activate mesh_utils
conda install -c conda-forge pymesh2
conda install -c pytorch pytorch=1.7.0 torchvision cudatoolkit=10.2
conda install -c conda-forge fvcore
conda install -c iopath iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
conda install jupyter
pip install -r requirements.txt
