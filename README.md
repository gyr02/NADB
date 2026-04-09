<h1 align="center"> Resolving Endpoint Underfitting in Diffusion Bridges via Noise Alignment (CVPR2026) </h1>
### | [Arxiv]() |

Official PyTorch implementation of **NADB**
## Installation

This code is developed with Python3, and we recommend PyTorch >=1.11. Install the dependencies with [Anaconda](https://www.anaconda.com/products/individual) and activate the environment `NADB` with
```bash
conda env create --file requirements.yaml python=3
conda activate NADB
```
## Data  Preparation
For restoration tasks, our code is built on I2SB, please follow the instructions in [I2SB](https://github.com/NVlabs/I2SB/edit/master/README.md#data-and-results).
## Training
To train an **NADB** on a single node, run
```bash
python train.py --name $NAME --n-gpu-per-node $N_GPU \
    --corrupt $CORRUPT --dataset-dir $DATA_DIR \
    --batch-size $BATCH --microbatch $MICRO_BATCH [--ot-ode] \
     --log-dir $LOG_DIR [--log-writer $LOGGER]
```
where `NAME` is the experiment ID (default: `CORRUPT`), `N_GPU` is the number of GPUs on each node, `DATA_DIR` is the path to the LMDB dataset.  `CORRUPT` can be one of the following restoration tasks:
- JPEG restoration: quality factor 5 or 10 (`jpeg-5`,`jpeg-10`)
- 4x Super-resolution: pool or bicubic filter (`sr4x-pool`,`sr4x-bicubic`)
- deblurring: uniform or Gaussian kernel  (`blur-uni`, `blur-gauss`)

Add `--ot-ode` for optionally training an OT-ODE model, _i.e.,_ the limit when the diffusion vanishes. By defualt, the model is discretized into 1000 steps; you can change it by adding `--interval $INTERVAL`.
Note that we initialize the network with [ADM](https://github.com/openai/guided-diffusion) ([256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)), which will be automatically downloaded to `data/` at first call.

To resume previous training from the checkpoint, add the flag `--ckpt $CKPT`.
## Hyperparameter
```bash
alpha=0.4
k=0.75
d=(1-alpha)/(2-alpha)
```
## Sampling

To sample from some checkpoint `$NAME` saved under `results/$NAME`, run

```bash
python sample.py --ckpt $NAME --n-gpu-per-node $N_GPU \
    --dataset-dir $DATA_DIR --batch-size $BATCH --use-fp16 \
    [--nfe $NFE] [--clip-denoise]
```
We set clip-denoise==None. 

PSNR, SSIM, and LPIS will also be automatically calculated.
## Evaluation

To evaluate the reconstruction images saved under `results/$NAME/$SAMPLE_DIR/`, run
```bash
python compute_metrices.py --ckpt $NAME --dataset-dir $DATA_DIR --sample-dir $SAMPLE_DIR
```
The FID computation is based on [`clean-fid`](https://github.com/GaParmar/clean-fid) package with `mode="legacy_pytorch"`.
## Acknowledgements

This code is developed heavily relying on [I2SB](https://github.com/NVlabs/I2SB). The code of dataloader in image translation tasks is drawn from [DDBM](https://github.com/alexzhou907/DDBM).
Thanks for these great projects. Please follow the license of the above open-source code.
