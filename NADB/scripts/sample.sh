
# change this variable to match your machine.
# For all the tasks, k=0.75, alpha=0.4
N_GPU=8

# JPEG restoration(jpeg-5,jpeg-10)
python sample.py --n-gpu-per-node $N_GPU  --ckpt jpeg-5 --use-mean --d "smallest" --nfe

# 4x super-resolution(sr4x-pool,sr4x-bicubic)
python sample.py --n-gpu-per-node $N_GPU  --ckpt sr4x-pool --use-mean --d "smallest" --nfe

# Deblurring(blur-uni,blur-gauss)
python sample.py --n-gpu-per-node $N_GPU  --ckpt blur-uni --use-mean --d "smallest" --nfe

# Image translation(edges2handbags)
python sample.py --n-gpu-per-node $N_GPU  --ckpt edges2handbags --use-mean --d 1 --nfe
#"--train-mean" can effectively improve the distortion loss metric.

# Image translation(edges2shoes)
#Do not use mean network
python sample.py --n-gpu-per-node $N_GPU  --ckpt edges2shoes  --d 1 --nfe


