# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm 
from PIL import Image

data = torch.load("/NADB/results/edges2handbags/samples_nfe119/recon.pt") #
tensor_imgs = data['arr']
output_dir ="/NADB/results/edges2handbags/samples_nfe119/images"


for i in tqdm(range(tensor_imgs.shape[0])):
    img_tensor = tensor_imgs[i].permute(1, 2, 0).cpu()
    
    img_np = (img_tensor.numpy() + 1) * 127.5  
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)  
    
  
    img_name = os.path.join(output_dir, f"image_{i:04d}.png")
    Image.fromarray(img_np).save(img_name)
    
    

print("finish")