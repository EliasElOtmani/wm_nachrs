import numpy as np 
import torch, os
from numpy import nan 

from CCM import CCM 
from DMTS import DMTS
from dmts_frontex import save_data

computer = "cpu"
dev_str = 'cpu'
cores = 6
num_workers = 4 * cores
enc = torch.float64
dev = torch.device(computer)

info = False
reject = False 
plot = False 

mod_prm = torch.as_tensor([.020, 0.007, .600, 2.2, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/26, 1, 1], device=dev, dtype=enc)
sim_prm = torch.as_tensor([5., 0.001, 1e-12, 1e-3, nan], device=dev, dtype=enc)
task_prm = [100, 1, 0.05, (0.2,0.3), (2, 2.1), (3.,3.1)]


Ae, Ap, As, Av = 169, 268, 709, 634
dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee,    wpe,     wse,     wes,     wvs,     wep,     wpp,     wsp,     wev,     wsv (10) :
    .136*Ae, .075*Ap, .045*As, .053*Ae, .04*Av, .17*Ae, .093*Ap, .0*As, .053*Ae, .001*As, 
    4.4, 4.8, 2.7, 1.9, # Ie_ext, Ip_ext, Is_ext, Iv_ext (5)
    0.2, .03*Ae, 25 # q, J_adp, sigma (3)
    
], device=dev, dtype=enc)

dist_amp = 0 # To be determined 
nic_amplitude = 0 # To be determined

def a5snp_transient(jobID):

    mod_prm = torch.as_tensor([.020, 0.007, .600, 2.2, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/26, 1, 1], device=dev, dtype=enc)
    sim_prm = torch.as_tensor([6., 0.001, 1e-12, 1e-3, nan], device=dev, dtype=enc)
    task_prm = [100, 2, 0.2, (0.2,0.3), (2, 2.1), (3.,3.1)]

    nic_amplitude = 0.75
    distractor_amplitude = 5

    dof_vip = dof.clone()
    dof_vip[17] -= 0.01*jobID
    '''
    ccm = CCM(dof_vip, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings_bloem = dmts.loadings/dmts.nb_trials

    ccm = CCM(dof_vip, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True, nic_normalization = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings_kamigaki = dmts.loadings/dmts.nb_trials

    toSave = np.array([encodings_bloem, encodings_kamigaki])

    save_data('a5snp_transient', 'a5snp_trans', toSave, jobID)
    '''

    ccm = CCM(dof_vip, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = False, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings = dmts.loadings/dmts.nb_trials

    toSave = np.array([encodings])
    save_data('a5snp_attention_reorientation_no_transient', 'encoding', toSave, jobID)


def nicotine_restoration(jobID):

    mod_prm = torch.as_tensor([.020, 0.007, .600, 2.2, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/26, 1, 1], device=dev, dtype=enc)
    sim_prm = torch.as_tensor([6., 0.001, 1e-12, 1e-3, nan], device=dev, dtype=enc)
    task_prm = [100, 2, 0.2, (0.2,0.3), (2, 2.1), (3.,3.1)]

    nic_amplitude = 0.75
    distractor_amplitude = 5

    dof[16] -= 0.01*jobID

    dof_vip = dof.clone()
    dof_vip_bloem = dof.clone()
    dof_vip[17] = 1.7 
    dof_vip_bloem[17] = 1.4

    ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings_bloem_healthy = dmts.loadings/dmts.nb_trials

    ccm = CCM(dof_vip_bloem, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings_bloem_a5snp = dmts.loadings/dmts.nb_trials

    ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True, nic_normalization = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings_kamigaki_healthy = dmts.loadings/dmts.nb_trials

    ccm = CCM(dof_vip, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True, nic_normalization = True)
    dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_amplitude = distractor_amplitude, distractor_timing = 1, distractor_duration = 0.05)
    encodings_kamigaki_a5snp = dmts.loadings/dmts.nb_trials

    toSave = np.array([encodings_bloem_healthy, encodings_bloem_a5snp, encodings_kamigaki_healthy, encodings_kamigaki_a5snp])

    save_data('nicotine_restoration', 'nicotine', toSave, jobID)
