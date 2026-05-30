
# change this variable to match your machine.
# For all the tasks, k=0.75, alpha=0.4
N_GPU=8

# JPEG restoration(jpeg-5,jpeg-10)
python train.py --n-gpu-per-node $N_GPU --train-mean --corrupt jpeg-5
python train.py --n-gpu-per-node $N_GPU --use-mean --corrupt jpeg-5

# 4x super-resolution(sr4x-pool,sr4x-bicubic)
python train.py --n-gpu-per-node $N_GPU --train-mean --corrupt sr4x-pool
python train.py --n-gpu-per-node $N_GPU --use-mean --corrupt sr4x-pool

# Deblurring(blur-uni,blur-gauss)
python train.py --n-gpu-per-node $N_GPU --train-mean --corrupt blur-uni
python train.py --n-gpu-per-node $N_GPU --use-mean --corrupt blur-uni

# Image translation(edges2handbags)
python train.py --n-gpu-per-node $N_GPU --train-mean --corrupt edges2handbags
python train.py --n-gpu-per-node $N_GPU --use-mean --corrupt edges2handbags --cond-x1
#"--train-mean" can effectively improve the distortion loss metric.

# Image translation(edges2shoes)
#Do not use mean network
python train.py --n-gpu-per-node $N_GPU  --corrupt edges2shoes --cond-x1


