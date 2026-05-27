# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from functools import partial
import torch
import torch.nn.functional as F
from .util import unsqueeze_xdim
import math
from ipdb import set_trace as debug
import torchvision.utils as vutils

class Diffusion():
    def __init__(self, opt):

        self.device = opt.device
        self.k=opt.k
        self.alpha=opt.alpha


    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample x_t and traing target , i.e. eq 5 snf eq 6 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        t = unsqueeze_xdim(step/1000, xdim).to(x1.device)
        k=self.k
        a=self.alpha
        ta=t**a
        xt=(1-ta)*x0+ta*(x1)
        I=torch.randn_like(xt)
        xt = xt + k*t*(1-t)*I
        label=(xt-x0)/ta
        return xt.detach(),label.detach()
   
    def p_posterior(self, nprev, n, x_n,x1, drift,d=d, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        a=self.alpha
        batch, *xdim = x_n.shape
        s = unsqueeze_xdim(torch.tensor(nprev/1000), xdim).to(x_n.device)
        t = unsqueeze_xdim(torch.tensor(n/1000), xdim).to(x_n.device)
        ta=t**a
        sa=s**a
        w=s/t
        if d=="smallest":
            d=(1-a)/(2-a)
        else:
            d=d
        if s<d:
            a=ta*(sa-1+w-w*ta)
            b=1-sa+w*ta
            c=sa-w*ta
            mean=a*drift+b*x_n+c*x1
        else:
            deta_t=ta-sa
            mean=x_n-drift*deta_t
        xt_prev =mean
        if  nprev > 0:
            k=self.k
            if s<d:
                var=(k*s*(1-s))**2-(k*t*(1-t)*w)**2
            else:
                var=(k*s*(1-s))**2-(k*t*(1-t)*sa/ta)**2
            if var< 0:
                print(s,t)
                print(var)
                1/0
            var=torch.sqrt(var)
            xt_prev = xt_prev + var * torch.randn_like(xt_prev)
        return xt_prev
    
    def ddpm_sampling(self, img_clean,steps, pred_drift,x1_to_x10, x1, d=d, ot_ode=False, log_steps=None, verbose=True):
        x1=x1_to_x10(x1.to(self.device))
        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0
       
        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
    
            assert prev_step < step, f"{prev_step=}, {step=}"
            
            drift = pred_drift(xt, step)
            xt = self.p_posterior(prev_step, step, xt,x1, drift,d=d, ot_ode=ot_ode)

            
            if prev_step in log_steps:
                pred_x0s.append(xt.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
