import numpy as np
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from CCM import CCM

computer = "cpu"
dev_str = 'cpu'
cores = 6
num_workers = 4 * cores
enc = torch.float64
dev = torch.device(computer)


mod_prm = torch.as_tensor([.020, .600, .8, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7.], device=dev, dtype=enc)
sim_prm = torch.as_tensor([200., .01, 1., 1e-12, 1e-3, nan], device=dev, dtype=enc)

Ae, Ap, As, Av = 172., 261., 757., 664.

dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10)
    # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10)
    3.9, 4.5, 3.6, 2.9, .45, # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .056*Ae, .001, .15 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)


plot = False 

ccm = CCM(dof, mod_prm, sim_prm)
tsr, stim = ccm.simulate()

print(tsr)
print('\n\n')
print(stim)
print(ccm.equilibria())



# So now we'll have to find a mode "task" that replaces the poisson input by 
# dirac (have to modify body of CCM) + modify parameters 
# for now get summary stats ! 