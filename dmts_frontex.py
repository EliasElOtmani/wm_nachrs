# Each function outputs the trial results for all populations & conditions for a given
# external current, in the following format :  
# dmts_<classic>_<Iext>.npy : np.array([VIP, PV, SOM, SOM & PV, All]) 

import numpy as np 
import torch, os
from numpy import nan 

from CCM import CCM 
from DMTS import DMTS

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


def save_data(output_dir, outputfile, data, jobID): 
	
	default_path = '/shared/projects/project_nicWM/data/'
	full_path = default_path + output_dir
	if not os.path.exists(full_path) : os.mkdir(full_path)
	
	with open(full_path + '/' + outputfile + '_' + str(jobID) + '.npy', 'wb') as f:
		np.save(f, data)
	
	print(data)

def classic_dmts(jobID):

	# Generates DMTS data and stores it

	array_length = 120
	middle = int(array_length/2)
	Istep = 0.025
	'''
	# VIP
	dof_vip = dof.clone()
	dof_vip[17] -= (middle - jobID)*Istep 	# jobID = 0 --> Iext -= 2.5

	ccm = CCM(dof_vip, mod_prm, sim_prm, STP = True, two_columns = False)
	dmts = DMTS(ccm, task_prm)
	load_vip = dmts.loadings/dmts.nb_trials
	delay_vip = dmts.maintenances/dmts.nb_trials

	# PV
	dof_pv = dof.clone()
	dof_pv[15] -= (middle - jobID)*Istep 	

	ccm = CCM(dof_pv, mod_prm, sim_prm, STP = True, two_columns = False)
	dmts = DMTS(ccm, task_prm)
	load_pv = dmts.loadings/dmts.nb_trials
	delay_pv = dmts.maintenances/dmts.nb_trials

	# SOM
	dof_som = dof.clone()
	dof_som[16] -= (middle - jobID)*Istep 	

	ccm = CCM(dof_som, mod_prm, sim_prm, STP = True, two_columns = False)
	dmts = DMTS(ccm, task_prm)
	load_som = dmts.loadings/dmts.nb_trials
	delay_som = dmts.maintenances/dmts.nb_trials

	# PV & SOM
	dof_pv_som = dof.clone()
	dof_pv_som[15] -= (middle - jobID)*Istep 	
	dof_pv_som[16] -= (middle - jobID)*Istep

	ccm = CCM(dof_pv_som, mod_prm, sim_prm, STP = True, two_columns = False)
	dmts = DMTS(ccm, task_prm)
	load_pv_som = dmts.loadings/dmts.nb_trials
	delay_pv_som = dmts.maintenances/dmts.nb_trials
	'''
	# ACh DEPLETION
	dof_ACh = dof.clone()
	dof_ACh[15] -= (middle - jobID)*(Istep/5) 	
	dof_ACh[16] -= (middle - jobID)*Istep
	dof_ACh[17] -= (middle - jobID)*Istep

	ccm = CCM(dof_ACh, mod_prm, sim_prm, STP = True, two_columns = False)
	dmts = DMTS(ccm, task_prm)
	load_ACh = dmts.loadings/dmts.nb_trials
	delay_ACh = dmts.maintenances/dmts.nb_trials

	#toSave = np.array([load_vip, delay_vip, load_pv, delay_pv, load_som, delay_som, load_pv_som, delay_pv_som, load_ACh, delay_ACh])

	#save_data('classic_dmts', 'dmts', toSave, jobID)
	toSave = np.array([load_ACh, delay_ACh])
	save_data('classic_dmts_Ach', 'dmts_ach', toSave, jobID)

def itrans_loading(jobID):

	duration = 0.002 + 0.002*jobID  	# jobID = 0-100
	intensities = [0.5 + 0.1*i for i in range(31)]

	dim_intensities = len(intensities)
	loadings = np.empty((dim_intensities))

	#task_prm = [100, 1, 0.05, (0.2,0.3), (2, 2.1), (3.,3.1)]
	task_prm_itrans = [elem for elem in task_prm]
	task_prm_itrans[2] = duration

	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = False, STP = True)

	for i in range(dim_intensities):
	    ccm.thalamus.Itrans = intensities[i]
	    dmts = DMTS(ccm, task_prm = task_prm_itrans)
	    loadings[i] = dmts.loadings/dmts.nb_trials

	save_data('detections', 'detection', loadings, jobID)


def pick_jadp(jobID):

	dof_jadp = dof.clone()
	dof_jadp[-2] = (0.01 + jobID*0.005)*Ae

	ccm = CCM(dof_jadp, mod_prm, sim_prm, equilibria = False, two_columns = False, STP = True)
	dmts = DMTS(ccm, task_prm = task_prm)
	maintenances = dmts.loadings/dmts.nb_trials
	save_data('pick_jadp', 'jadp', maintenances, jobID)

def pick_distractor_amplitude(jobID):

	dist_amp = 1.5 + 0.025*jobID
	
	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True)
	dmts = DMTS(ccm, task_prm = task_prm, distractor = True, distractor_amplitude = dist_amp, distractor_duration = 0.2)
	maintenances = dmts.maintenances/dmts.nb_trials
	save_data('pick_distractor_amp_pdr1', 'dAmp', maintenances, jobID)


def distracted_dmts(jobID):

	#array_length = 200
	#middle = int(array_length/2)
	Istep = 0.01
	dist_amp = 0 # 2.2
	
	jobID += 100
	#if jobID < middle : jobID -= 100
	#else : jobID += 100
	'''
	# VIP
	dof_vip = dof.clone()
	dof_vip[17] -= (middle - jobID)*Istep 	# jobID = 0 --> Iext -= 2.5

	ccm = CCM(dof_vip, mod_prm, sim_prm, STP = True, two_columns = True)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_amplitude = dist_amp)
	load_vip = dmts.loadings/dmts.nb_trials
	delay_vip = dmts.maintenances/dmts.nb_trials

	# PV
	dof_pv = dof.clone()
	dof_pv[15] -= (middle - jobID)*Istep 	

	ccm = CCM(dof_pv, mod_prm, sim_prm, STP = True, two_columns = True)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_amplitude = dist_amp)
	load_pv = dmts.loadings/dmts.nb_trials
	delay_pv = dmts.maintenances/dmts.nb_trials

	# SOM
	dof_som = dof.clone()
	dof_som[16] -= (middle - jobID)*Istep 	

	ccm = CCM(dof_som, mod_prm, sim_prm, STP = True, two_columns = True)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_amplitude = dist_amp)
	load_som = dmts.loadings/dmts.nb_trials
	delay_som = dmts.maintenances/dmts.nb_trials
	'''
	# PV & SOM
	dof_pv_som = dof.clone()
	#dof_pv_som[15] -= (middle - jobID)*Istep 	
	#dof_pv_som[16] -= (middle - jobID)*Istep

	dof_pv_som[15] += jobID*Istep
	dof_pv_som[16] += jobID*Istep

	ccm = CCM(dof_pv_som, mod_prm, sim_prm, STP = True, two_columns = True)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_amplitude = dist_amp)
	load_pv_som = dmts.loadings/dmts.nb_trials
	delay_pv_som = dmts.maintenances/dmts.nb_trials
	'''
	# ACh DEPLETION
	dof_ACh = dof.clone()
	dof_ACh[15] -= (middle - jobID)*(Istep/5) 	
	dof_ACh[16] -= (middle - jobID)*Istep
	dof_ACh[17] -= (middle - jobID)*Istep

	ccm = CCM(dof_ACh, mod_prm, sim_prm, STP = True, two_columns = True)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_amplitude = 1)
	load_ACh = dmts.loadings/dmts.nb_trials
	delay_ACh = dmts.maintenances/dmts.nb_trials

	toSave = np.array([load_vip, delay_vip, load_pv, delay_pv, load_som, delay_som, load_pv_som, delay_pv_som, load_ACh, delay_ACh])

	save_data('distracted_dmts_pdr1_amp0', 'ddmts', toSave, jobID)
	'''
	
	toSave = np.array([load_pv_som, delay_pv_som])
	save_data('distracted_dmts_pdr1_amp0_extremes', 'ddmts', toSave, jobID)

def nicTrans_dmts(jobID):	# Tries different values of nic_trans_amplitude on the 'healthy' model 
	# Generates DMTS data and stores it
	
	task_prm = [100, 2, 0.2, (0.2,0.3), (2,2.1), (3,3.1)]
	sim_prm = torch.as_tensor([6., 0.001, 1e-12, 1e-3, nan], device = dev, dtype = enc)

	#nic_amplitude = 0.1 + jobID*0.025
	nic_amplitude = jobID*0.025

	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	loadings = dmts.loadings/dmts.nb_trials

	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	loadings_kam = dmts.loadings/dmts.nb_trials

	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, nic_normalization = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	loadings_norm = dmts.loadings/dmts.nb_trials

	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True, nic_normalization = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	loadings_kam_norm = dmts.loadings/dmts.nb_trials

	toSave = np.array([loadings, loadings_kam, loadings_norm, loadings_kam_norm])
	save_data('nicTrans_dmts_3', 'nicTrans_dmts', toSave, jobID)
	#print(toSave)

def nicotine_restoration(jobID):

	task_prm = [100, 2, 0.2, (0.2,0.3), (2,2.1), (3,3.1)]
	sim_prm = torch.as_tensor([6., 0.001, 1e-12, 1e-3, nan], device = dev, dtype = enc)

	dof[16] += 0.01*jobID

	dof_a5snp = dof.clone()
	dof_a5snp[17] -= TO_BE_DETERMINED
	
	nic_amplitude = 0.75

	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True)
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	maintenances_healthy_bloem = dmts.maintenances/dmts.nb_trials

	ccm = CCM(dof_a5snp, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	maintenances_a5snp_bloem = dmts.maintenances/dmts.nb_trials
	
	ccm = CCM(dof, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True, nic_normalization = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	maintenances_healthy_kamigaki = dmts.maintenances/dmts.nb_trials

	ccm = CCM(dof_a5snp, mod_prm, sim_prm, equilibria = False, two_columns = True, STP = True, kamigaki = True, nic_normalization = True )
	dmts = DMTS(ccm, task_prm = task_prm, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2, distractor = True, distractor_timing = 1, distractor_amplitude=5, distractor_duration=0.05)
	maintenances_a5snp_kamigaki = dmts.maintenances/dmts.nb_trials

	toSave = np.array([maintenances_healthy_bloem, maintenances_a5snp_bloem, maintenances_healthy_kamigaki, maintenances_a5snp_kamigaki])
	save_data()


def nicotinic_dmts(jobID, nic_normalization = False, kamigaki = False):

	array_length = 120
	middle = int(array_length/2)
	Istep = 0.025

	sim_prm = torch.as_tensor([6., 0.001, 1e-12, 1e-3, nan], device=dev, dtype=enc)
	task_prm = [100, 2, 0.05, (0.2,0.3), (2, 2.1), (3.,3.1)] # CHANGE CUE DURATION & AMPLITUDE !

	# VIP
	dof_vip = dof.clone()
	dof_vip[17] -= (middle - jobID)*Istep 	# jobID = 0 --> Iext -= 2.5

	ccm = CCM(dof_vip, mod_prm, sim_prm, STP = True, two_columns = True, nic_normalization = nic_normalization, kamigaki = kamigaki)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_timing = 1, distractor_amplitude = dist_amp, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2)
	load_vip = dmts.loadings/dmts.nb_trials
	delay_vip = dmts.maintenances/dmts.nb_trials

	# PV
	dof_pv = dof.clone()
	dof_pv[15] -= (middle - jobID)*Istep 	

	ccm = CCM(dof_pv, mod_prm, sim_prm, STP = True, two_columns = True, nic_normalization = nic_normalization, kamigaki = kamigaki)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_timing = 1, distractor_amplitude = dist_amp, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2)
	load_pv = dmts.loadings/dmts.nb_trials
	delay_pv = dmts.maintenances/dmts.nb_trials

	# SOM
	dof_som = dof.clone()
	dof_som[16] -= (middle - jobID)*Istep 	

	ccm = CCM(dof_som, mod_prm, sim_prm, STP = True, two_columns = True, nic_normalization = nic_normalization, kamigaki = kamigaki)
	load_som = dmts.loadings/dmts.nb_trials
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_timing = 1, distractor_amplitude = dist_amp, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2)
	delay_som = dmts.maintenances/dmts.nb_trials

	# PV & SOM
	dof_pv_som = dof.clone()
	dof_pv_som[15] -= (middle - jobID)*Istep 	
	dof_pv_som[16] -= (middle - jobID)*Istep

	ccm = CCM(dof_pv_som, mod_prm, sim_prm, STP = True, two_columns = True, nic_normalization = nic_normalization, kamigaki = kamigaki)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_timing = 1, distractor_amplitude = dist_amp, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2)
	load_pv_som = dmts.loadings/dmts.nb_trials
	delay_pv_som = dmts.maintenances/dmts.nb_trials

	# ACh DEPLETION
	dof_ACh = dof.clone()
	dof_ACh[15] -= (middle - jobID)*(Istep/5) 	
	dof_ACh[16] -= (middle - jobID)*Istep
	dof_ACh[17] -= (middle - jobID)*Istep

	ccm = CCM(dof_ACh, mod_prm, sim_prm, STP = True, two_columns = True, nic_normalization = nic_normalization, kamigaki = kamigaki)
	dmts = DMTS(ccm, task_prm, distractor = True, distractor_timing = 1, distractor_amplitude = dist_amp, nic_trans = True, nic_trans_amplitude = nic_amplitude, nic_trans_timing = 2)
	load_ACh = dmts.loadings/dmts.nb_trials
	delay_ACh = dmts.maintenances/dmts.nb_trials

	toSave = np.array([load_vip, delay_vip, load_pv, delay_pv, load_som, delay_som, load_pv_som, delay_pv_som])

	save_data('nicotinic_dmts', 'nicdmts', toSave, jobID)


#if not os.path.exists('output_dir'): os.mkdir('output_dir')
