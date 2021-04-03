#	C:\Users\elias\Desktop\CogSci\Internship\Code

import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from CCM import CCM
from Intracol_WTA import Intracol_WTA
from DMTS import DMTS

# Convert tensor tens to np array : tens = tens.numpy()



# SEE the self.reject and aborted if len S < 3 : gotta find a way to implement it befor the simulation has been created

computer = "cpu"
dev_str = 'cpu'
cores = 6
num_workers = 4 * cores
enc = torch.float64
dev = torch.device(computer)


#mod_prm = torch.as_tensor([.020, .600, .8, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7.], device=dev, dtype=enc)


#########################################################################################################################################################
# FREE PARAMETERS #
###################


mod_prm = torch.as_tensor([.020, .600, 4.5, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/45, 1], device=dev, dtype=enc)		# refractory period : from 0.0025 to 0.01
sim_prm = torch.as_tensor([20, .005, 1e-12, 1e-3, nan], device=dev, dtype=enc)

Ae, Ap, As, Av = 169, 268, 709, 634
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee, 	  wpe,     wse,     wes,     wvs,     wep,     wpp,     wsp,     wev,     wsv (10) :
    .136*Ae, .101*Ap, .002*As, .077*Ae, .048*Av, .112*Ae, .093*Ap, .0*As, .041*Ae, .001*As, 
    3.9, 4.5, 3.6, 2.9, # Ie_ext, Ip_ext, Is_ext, Iv_ext (5)
    0.6, .063*Ae, .02 # q, J_adp, sigma (3)
    
], device=dev, dtype=enc)


info = True
reject = False
plot = False 

############################################################################################################################################################

ccm = CCM(dof, mod_prm, sim_prm, equilibria = False)

ccm.simulate(reject = reject, dmts = True, info = info)
sim = ccm.simulations[0]
tsr, stim = sim.traces, sim.stimuli
eqs = ccm.S
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
plt.plot(tsr[1,:], color = 'green', linewidth = 1)
plt.plot(tsr[0,:], color = 'red', linewidth = 3)
plt.plot(tsr[2,:], color = 'orange', linewidth = 1)
#plt.plot(tsr[3,:], color = 'orange', linewidth = 1)
#plt.plot(tsr[4,:], color = 'lightgreen', linewidth = 1)
plt.show()


'''
dmts = DMTS(ccm)
dmts.print_stats()
dmts.plot_trials()
'''