#	C:\Users\elias\Desktop\CogSci\Internship\Code

import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from CCM import CCM
from DMTS import DMTS

# Convert tensor tens to np array : tens = tens.numpy()


computer = "cpu"
dev_str = 'cpu'
cores = 6
num_workers = 4 * cores
enc = torch.float64
dev = torch.device(computer)


#mod_prm = torch.as_tensor([.020, .600, .8, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7.], device=dev, dtype=enc)
mod_prm = torch.as_tensor([.020, .600, 4.5, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/45], device=dev, dtype=enc)
sim_prm = torch.as_tensor([200, .01, 1., 1e-12, 1e-3, nan], device=dev, dtype=enc)


#########################################################################################################################################################
# FREE PARAMETERS #
###################


# MROOY THESIS AND FIRST PREPRINT	/!\ Wvs taken as the value of the second wep in the thesis.    CHECK WSP
################################# 

Ae, Ap, As, Av = 169, 268, 709, 634
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee, 	  wpe,     wse,     wes,     wvs,     wep,     wpp,     wsp,     wev,     wsv (10) :
    .136*Ae, .101*Ap, .002*As, .077*Ae, .048*Av, .112*Ae, .093*Ap, .0*As, .041*Ae, .001*As, 
    3.9, 4.5, 3.6, 2.9, 4.5, # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .0598*Ae, .01, 1/45 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)

# Modifications : I_trans (.45 initially), usf (.15 initially), sigma (.001 initially)
#NB : For I_tran, the threshold seems to be between 2 and 3 (2 not enough to elicit oscillations)




mod_prm = torch.as_tensor([.020, .600, 4.5, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/45, 0.005], device=dev, dtype=enc)		# refractory period : from 0.0025 to 0.01
sim_prm = torch.as_tensor([200, .01, 1., 1e-12, 1e-3, nan], device=dev, dtype=enc)

Ae, Ap, As, Av = 169, 268, 709, 634
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee, 	  wpe,     wse,     wes,     wvs,     wep,     wpp,     wsp,     wev,     wsv (10) :
    .136*Ae, .101*Ap, .002*As, .077*Ae, .048*Av, .112*Ae, .093*Ap, .0*As, .041*Ae, .001*As, 
    3.9, 4.5, 3.6, 2.9, # Ie_ext, Ip_ext, Is_ext, Iv_ext (5)
    0.8, .0598*Ae, .02 # q, J_adp, sigma (3)
    
], device=dev, dtype=enc)

############################################################################################################################################################


info = True
reject = True 
plot = False 

ccm = CCM(dof, mod_prm, sim_prm)

ccm.simulate(dmts = False)
sim = ccm.simulations[0]
tsr, stim = sim.traces, sim.stimuli
eqs = sim.S
res = sim.postproc()
tsr = tsr.numpy()

if info:
	print('\n Tested parameters: \n\n', dof,
		  '\n\n Simulation window [s]: ', sim.window, ' ; Time resolution [s]:', sim.dt, ' ; Refractory period [dt]: ', sim.tr,
		  '\n\n Number of stimuli: ', len(stim), ' ; Data shape: ', tsr.shape,
		  '\n\n Number of equilibria: ', len(eqs),
		  '\n\n Equilibria: \n\n', np.sort(eqs, 0),
		  '\n\n Summary statistics of simulated data: \n\n', res, '\n')

#print('\n\n', np.sort(ccm.S, 0)[1])


#print(ccm.equilibria())


fig, ax = plt.subplots()
plt.plot(tsr[0,:], color = 'red', linewidth = 2)
plt.show()

'''
dmts = DMTS(ccm)
dmts.print_stats()
dmts.plot_trials()
'''
































'''
Ae, Ap, As, Av = 172., 261., 757., 664.

dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10) :
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, 
    3.9, 4.5, 3.6, 2.9, .45, # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .056*Ae, .001, .15 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)


# PREPRINT
#dof = torch.as_tensor([Ae,Ap,As,Av,.137, .101, .002, .077, .048, .113, .093, .024, .004, .042, .001, .056, .015, 3.9, 4.5, 3.6, 2.9, .5], device=dev, dtype=enc)
dof = torch.as_tensor([5.4093e+02, 1.2520e+02, 5.6639e+02, 3.4478e+02, 3.8753e+01, 1.0218e+00,
        1.7771e+01, 6.9165e+01, 6.1233e+01, 2.6039e+01, 3.1769e+01, 1.7762e+01,
        2.0576e+01, 1.1006e+01, 2.9551e+00, 4.3278e+00, 2.9319e+00, 7.4019e-01,
        2.3594e+01, 15, 1.3175e-02, 4.2953e-02], device = dev, dtype = enc)
#6.4881e+00 original Jadp


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


####################################


# MRooy's thesis with apparent normalization error
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10)
    3.9, 4.5, 3.6, 2.9, .45, # Ie_ext, Ip_ext, Is_ext, Iv_ext, I_trans (5)
    .056*Ae, .001, .15 # J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)


#FROM THOMAS' "MODEL INFERENCE"

Ae, Ap, As, Av = 172., 261., 757., 664. # According to pre-print [Rooy, 2018].
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    .137*Ae, .101*Ae, .002*Ae, .077*As, .048*As, .113*Ap, .093*Ap, .004*Ap, .042*Av, .001*Av, # wee, wpe, wse, wes, wvs, wep, wpp, wsp, wev, wsv (10)
    3.9/Ae, 4.5/Ap, 3.6/Av, 2.9/As, # Ie_ext, Ip_ext, Is_ext, Iv_ext (4)
    1., .056*Ae, .005, 1/45. # I_trans, J_adp, sigma, frequency of ultra-slow stimuli ('usf' : float)[Hz] (3)
    
], device=dev, dtype=enc)
'''


'''
ccm = CCM(dof, mod_prm = mod_prm, sim_prm = sim_prm, info = True)
tsr, stim = ccm.simulate()
#tsr = tsr.numpy() 


eqs = ccm.equilibria()
critic = torch.as_tensor(np.sort(eqs, 0)[1], device=dev, dtype=enc)
res = ccm.postproc(tsr, eqs, critic, info, reject)
'''