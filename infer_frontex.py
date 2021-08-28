def infer(job_id):

	import numpy as np 
	import matplotlib.pyplot as plt
	import torch

	from Simulation import Simulation
	from CCM import CCM 

	from numpy import nan

	def model(theta):

		mod_prm = torch.as_tensor([.020, .007, .600, 2, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/26, 0.008, 0.002], device=dev, dtype=enc)		# refractory period : from 0.0025 to 0.01
		sim_prm = torch.as_tensor([10, .005, 1e-12, 1e-3, nan], device=dev, dtype=enc)

		Wee, Wpe, Wse = theta[:3]
		Wes, Wvs = theta[3:5]
		Wep, Wpp, Wsp = theta[5:8]
		Wev, Wsv = theta[8:10]
		Ie_ext, Ip_ext, Is_ext, Iv_ext = theta[10:14]
		Tref_PYR, Tref_INs = mod_prm[14], mod_prm[15]	

		amplitudes = torch.as_tensor([1/Tref_PYR,1/Tref_INs,1/Tref_INs,1/Tref_INs])
		q = 0.5
		sigma = 0.1
		q_theta_jadp_sigma = torch.as_tensor([q, theta[-1], sigma])
		dof = torch.cat((amplitudes, theta[:-1], q_theta_jadp_sigma), 0)

		# /!\ Sigma was at the last position, q antépénultième ! 

		ccm = CCM(dof, mod_prm, sim_prm, equilibria = False)
		ccm.simulate(reject = False, dmts = False, info = False)
		sim = ccm.simulations[0]

		return sim.postproc()


	computer = "cpu"
	enc = torch.float64
	dev = torch.device(computer)

	dim = 15 # Number of degrees of freedom of the simulator [w_xy (10), I_ext-x (4), J_adp]

	# PRIORS
	mW, mI, mP = [.1 for k in range(10)], [1 for k in range(4)], [3]
	MW, MI, MP = [75. for k in range(10)], [10. for k in range(4)], [15] 
	priors_min, priors_max = np.concatenate((mW, mI, mP)), np.concatenate((MW, MI, MP))
	priors = np.array([priors_min, priors_max])

	prior_min = torch.as_tensor(priors_min, device=dev, dtype=torch.float64)
	prior_max = torch.as_tensor(priors_max, device=dev, dtype=torch.float64)


	#theta = np.random.uniform(low = priors_min, high = priors_max, size = (2000,17))

	theta = np.empty([2, 15])

	for params in theta :

		params[4] = np.random.uniform(low = priors_min[4], high = priors_max[4])				# Wvs
		params[6] = np.random.uniform(low = priors_min[6], high = priors_max[6]) 				# Wpp
		params[1] = np.random.uniform(low = 0.5*params[6], high = params[6])					# 0.5*Wpp < Wpe < Wpp
		params[7] = np.random.uniform(low = priors_min[7], high = 0.5*params[6])				# Wsp < 0.5*Wpp

		params[9] = np.random.uniform(low = (1/1.25)*params[7], high = 4*params[7]) 			# (1/1.25)*Wsp < Wsv < 4*Wsp  
		params[2] = np.random.uniform(low = priors_min[2], high = min(params[1], params[9]))	# Wse < Wsv, Wse < Wpe

		params[0] = np.random.uniform(low = priors_min[0], high = priors_max[0])				# Wee
		params[5] = np.random.uniform(low = params[0], high = priors_max[5])					# Wep > Wee
		params[8] = np.random.uniform(low = params[0], high = params[5])						# Wee < Wev < Wep
		params[3] = np.random.uniform(low = params[0], high = params[5])						# Wee < Wes < Wep
		params[11:] = np.random.uniform(low = priors_min[11:], high = priors_max[11:]) 			# Iext except Ie_ext, Jadp
		params[10] = np.random.uniform(low = max(params[11:-1]), high = priors_max[10])			# Ie_ext


	theta = torch.as_tensor(theta)

	x = list(map(model, theta))
	x = np.vstack([params.numpy() for params in x])

	toSave = np.hstack((x, theta))

	#file_id = [str(np.random.randint(0,9)) for i in range(10)]
	#file_id = "".join(file_id)

#	with open('/shared/projects/project_nicWM/data/round1/x_theta_' + str(job_id) + '.np', 'wb') as h:
#		np.save(h, toSave)

	with open('C:/Users/elias/Desktop/CogSci/Internship/Code/output_data/' + str(job_id) + '.npy', 'wb') as h:
		np.save(h, toSave)

