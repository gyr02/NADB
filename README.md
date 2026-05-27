<h1 align="center"> Resolving Endpoint Underfitting in Diffusion Bridges via Noise Alignment (CVPR 2026) </h1>

Official PyTorch implementation of **NADB**

### | [Arxiv]() |
We updated the hyperparameters in image translation, please see [`scripts/train.sh`](https://github.com/gyr02/NADB/scripts/train.sh) and 
[`scripts/sample.sh`](https://github.com/gyr02/NADB/scripts/sample.sh).


## Installation

This code is developed with Python3, and we recommend PyTorch >=1.11. Install the dependencies with [Anaconda](https://www.anaconda.com/products/individual) and activate the environment `NADB` with
```bash
conda env create --file environments.yaml python=3
conda activate NADB
```
## Data  Preparation
For restoration tasks, our code is built on I2SB, please follow the instructions in [I2SB](https://github.com/NVlabs/I2SB/edit/master/README.md#data-and-results).
## Training
To train  **NADB** on a single node, run
```bash
python train.py --name $NAME --n-gpu-per-node $N_GPU \
    --corrupt $CORRUPT --dataset-dir $DATA_DIR \
     --log-dir $LOG_DIR [--log-writer $LOGGER]\
     --train-mean \
     --use-mean
```
where `NAME` is the experiment ID (default: `CORRUPT`), 
`N_GPU` is the number of GPUs on each node, 
`DATA_DIR` is the path to the LMDB dataset.  
`CORRUPT` can be one of the following restoration tasks:
- JPEG restoration: quality factor 5 or 10 (`jpeg-5`,`jpeg-10`)
- 4x Super-resolution: pool or bicubic filter (`sr4x-pool`,`sr4x-bicubic`)
- deblurring: uniform or Gaussian kernel  (`blur-uni`, `blur-gauss`)

and image translation tasks:
- image trasnlation: (`edges2handbags`,`edges2shoes`)

For some tasks, we need to first use `train-mean` to train mean network and then use `use-mean`.

To resume previous training from the checkpoint, add the flag `--ckpt $CKPT`.

For specific parameter settings, please refer to  [`scripts/train.sh`](https://github.com/gyr02/NADB/scripts/train.sh) .

## Sampling

To sample from some checkpoint `$NAME` saved under `results/$NAME`, run

```bash
python sample.py --ckpt $NAME --n-gpu-per-node $N_GPU \
    --dataset-dir $DATA_DIR --batch-size $BATCH --use-fp16 \
    [--nfe $NFE] [--clip-denoise]
```
We set clip-denoise==None. 

PSNR, SSIM, and LPIPS will also be automatically calculated.
## Evaluation

To evaluate the reconstruction images saved under `results/$NAME/$SAMPLE_DIR/`, run
```bash
python compute_metrices.py --ckpt $NAME --dataset-dir $DATA_DIR --sample-dir $SAMPLE_DIR
```
The FID computation is based on [`clean-fid`](https://github.com/GaParmar/clean-fid) package with `mode="legacy_pytorch"`.

To evaluate FID on image translation tasks, please run
```bash
python torgb.py
python fid.py
```

## Hyperparameter
```bash
alpha=0.4
k=0.75
d=(1-alpha)/(2-alpha) or 1
w=t_{i-1}/t_{i}
```

## Licenses

This code is developed heavily relying on [I2SB](https://github.com/NVlabs/I2SB). The code of dataloader in image translation tasks is drawn from [DDBM](https://github.com/alexzhou907/DDBM).
Thanks for these great projects. Please follow the licenses of the above open-source code.

## Acknowledgements
Thank you,  Zhang Zicheng, for your strong support for this work. 
