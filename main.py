import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from CCM import CCM

# Convert tensor tens to np array : tens = tens.numpy()


computer = "cpu"
dev_str = 'cpu'
cores = 6
num_workers = 4 * cores
enc = torch.float64
dev = torch.device(computer)


mod_prm = torch.as_tensor([.020, .600, .8, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7.], device=dev, dtype=enc)
sim_prm = torch.as_tensor([20., .01, 1., 1e-12, 1e-3, nan], device=dev, dtype=enc)

Ae, Ap, As, Av = 172., 261., 757., 664.
'''
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10) :
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, 
    3.9, 4.5, 3.6, 2.9, .45, # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .056*Ae, .001, .15 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)
'''
'''
# PREPRINT
#dof = torch.as_tensor([Ae,Ap,As,Av,.137, .101, .002, .077, .048, .113, .093, .024, .004, .042, .001, .056, .015, 3.9, 4.5, 3.6, 2.9, .5], device=dev, dtype=enc)
dof = torch.as_tensor([5.4093e+02, 1.2520e+02, 5.6639e+02, 3.4478e+02, 3.8753e+01, 1.0218e+00,
        1.7771e+01, 6.9165e+01, 6.1233e+01, 2.6039e+01, 3.1769e+01, 1.7762e+01,
        2.0576e+01, 1.1006e+01, 2.9551e+00, 4.3278e+00, 2.9319e+00, 7.4019e-01,
        2.3594e+01, 15, 1.3175e-02, 4.2953e-02], device = dev, dtype = enc)
#6.4881e+00 original Jadp
'''

dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10)
    3.9, 4.5, 3.6, 2.9, 1., # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .056*Ae, .002, 1/26.1 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=torch.float64)

#Still from MODEL ANALYSIS, probably the same as the first one
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10)
    3.9, 4.5, 3.6, 2.9, .45, # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .056*Ae, .001, .05 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)

# /!\ Initial value of usf = .15 but getting 4 equilibria, 
# getting 3 for .05


info = True
reject = True 
plot = False 

ccm = CCM(dof, mod_prm, sim_prm, info = True)
tsr, stim = ccm.simulate()
#tsr = tsr.numpy() 


eqs = ccm.equilibria()
critic = torch.as_tensor(np.sort(eqs, 0)[1], device=dev, dtype=enc)
res = ccm.postproc(tsr, eqs, critic, info, reject)

tsr = tsr.numpy()

if info:
	print('\n Tested parameters: \n\n', dof,
		  '\n\n Simulation window [s]: ', ccm.window, ' ; Time resolution [s]:', ccm.dt, ' ; Refractory period [dt]: ', ccm.tr,
		  '\n\n Number of stimuli: ', len(stim), ' ; Data shape: ', tsr.shape,
		  '\n\n Number of equilibria: ', len(eqs),
		  '\n\n Equilibria: \n\n', np.sort(eqs, 0),
		  '\n\n Summary statistics of simulated data: \n\n', res, '\n')






print(np.shape(tsr))
print('\n\n')
print(np.shape(stim))
#print(ccm.equilibria())

fig, ax = plt.subplots()
plt.plot(tsr[0,:], color = 'red', linewidth = 2)
plt.show()




# So now we'll have to find a mode "task" that replaces the poisson input by 
# dirac (have to modify body of CCM) + modify parameters 
# for now get summary stats ! 