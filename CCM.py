#!/usr/bin/env python

__author__ = "Elias El Otmani"
__credits__ = "Thomas Fontaine"
__email__ = "elias.el.otmani@ens.psl.eu"

# An "instance" of CCM stands for one or two cortical columns with defined parameters. 
# Once declared at the instanciation, the latter only change if explicitly modified 
# Conversely, simulation parameters are easily re-declared for each simulation.

import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
 
from NeuralPop import NeuralPop
from Simulation import Simulation 


# DEFAULT PARAMETERS

mod_prm = torch.as_tensor([.020, .007, .600, 3, 0., 1.9, 2.6, 1.5, 1.2, 7., 7., 7., 7., 1/26, 1, 1], device=dev, dtype=enc)		# refractory period : from 0.0025 to 0.01
sim_prm = torch.as_tensor([5, .001, 1e-12, 1e-3, nan], device=dev, dtype=enc)

dof = torch.as_tensor([
    
    Ae, Ap, As, Av, # Ae, Ap, As, Av (4)
    # wee,    wpe,     wse,     wes,     wvs,     wep,     wpp,     wsp,     wev,     wsv (10) :
    .136*Ae, .075*Ap, .045*As, .053*Ae, .04*Av, .17*Ae, .093*Ap, .0*As, .053*Ae, .001*As, 
    4.4, 4.8, 2.7, 1.9, # Ie_ext, Ip_ext, Is_ext, Iv_ext (5)
    0.2, .03*Ae, 25 # q, J_adp, sigma (3)
    
], device=dev, dtype=enc)

# CCM 

class CCM():

	def __init__(self, dof = dof, mod_prm = mod_prm, sim_prm = None, equilibria = False, reject = False, computer = "cpu", enc = torch.float64, STP = True, two_columns = False, pdr = 0.5, notebook = None, kamigaki = False, nic_normalization = False):

		# TEMPORARY, find a way to better deal with this
		self.notebook = notebook

		self.dev = torch.device(computer)
		self.atol, self.rtol = 1e-12, 1e-3 

		self.sim_prm = sim_prm
		self.mod_prm = mod_prm
		self.dof = dof

		self.reject = reject

		self.simulations = []

		self.usf = mod_prm[13] # Frequency of ultra-slow fluctuations

		self.STP = STP
		self.two_columns = two_columns
		self.nic_normalization = nic_normalization
		self.kamigaki = kamigaki

		if not two_columns : self.thalamus, self.populations = self.grow_single_column(mod_prm, dof)
		else : self.thalamus, self.thalamus2, self.cholinergic_nuclei, self.populations = self.grow_two_columns(mod_prm, dof, pdr)

		self.dim = len(self.populations)

	### MODEL 
		
		# Sources of stochasticity.
		self.poisson = torch.distributions.poisson.Poisson(torch.tensor([self.usf])) # Poisson distribution of stimuli. 
		self.n = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # Normal distribution of neural noise.


	# Short-hand indices for tensor-based data storage [1](int)
		self.RE, self.RP, self.RS, self.RV, self.dRE, self.dRP, self.dRS, self.dRV, self.TT = 0, 1, 2, 3, 4, 5, 6, 7, 8
		self.INE_E, self.INE_P, self.INE_S, self.INE_V, self.INS_E, self.INS_P, self.INS_S, self.INS_V = 9, 10, 11, 12, 13, 14, 15, 16
		self.IND_E, self.IE_ADP = 17, 18


	# Compute equilibria as soon as instantiated : 
		self.hidden_eq = False # Hidden equilibria, supra-saturation dynamics
		if equilibria : 
			self.S = self.equilibria()
			try : self.critic = self.S[1]
			except : self.critic = torch.as_tensor([20 for i in range(self.dim)], device = self.dev, dtype = torch.float64)
		else : 
			self.S = [None]
			self.critic = torch.as_tensor([20 for i in range(self.dim)], device = self.dev, dtype = torch.float64)


	def simulate(self, sim_prm = sim_prm, info=False, plot=False, dmts = False, cue_timings = [1], Itrans_duration = 0.05, reject = None, distractor = False, distractor_timing = 2, distractor_duration = 0.05, distractor_amplitude = 1, nic_trans = False, nic_trans_timing = 1.5, nic_trans_duration = 1, nic_trans_amplitude = 0.5): # Maybe put reject in sim_prm
		"""
		
		Returns raw simulated data corresponding to time-evolutions of CCM's latent variables. 
		
		Entries:
		-------
		
			plot = bool : option parameter (default is False) conditioning output's dimensionality.
		
		Outputs:
		-------
		
		When plot is set to True:
			
			tsr = torch.Tensor(size=(17, _), device=dev, dtype=enc) : data tensor storing time-evolutions of all latent variables.
			stim = torch.Tensor(size=_, device=dev, dtype=enc) : data tensor storing stimuli time-events. 
			
		When plot is set to False:
		
			tsr = torch.Tensor(size=(9, _), device=dev, dtype=enc) : data tensor storing time-evolutions of neural populations' rates and their derivatives.
			
		"""

	### SIMULATION PARAMETERS
		if sim_prm is None : sim_prm = self.sim_prm
		if sim_prm is None : 
			print('No simulation parameters declared')
			return 

	### ABORT 
		if reject is None : reject = self.reject
		if reject and len(self.S) < 3:
			if info : print('Model not bistable')
			return Simulation(None, None, sim_prm, self.mod_prm, self.dof, self.critic, aborted = True)

		window = sim_prm[0].item() # Stimulation window [s](float).
		dt = sim_prm[1].item() # Time resolution [s](float).
		#atol, rtol = sim_prm[2].item(), sim_prm[3].item() # Absolute and relative tolerances for float comparison.
		
		if torch.isnan(sim_prm[4]):
			torch.manual_seed(time.time())
		else:
			torch.manual_seed(sim_prm[4])
		
		#smin = round(3 * max(self.tau, self.tau_adp) / dt) # Starting time for stimulation window [1](int).
		smin = 10 			# CORRECT THAT
		smax = round(window / dt) # End of stimulation window [1](int).
		N = smin + smax # Total number of time-points [1](int)

		if distractor_amplitude is not None and self.two_columns : self.thalamus2.Itrans = distractor_amplitude*100 
		if nic_trans_amplitude is not None and self.two_columns : self.cholinergic_nuclei.Itrans = nic_trans_amplitude*100

	### SIMULATION 

		tsr = torch.empty((self.dim+1, N), device=self.dev, dtype=torch.float64)
		stim = []

		Itrans_start = - Itrans_duration/dt
		distractor_start = - distractor_duration/dt
		nic_trans_start = - nic_trans_duration/dt 
	
		for k in range(N):
			
			# TRANSIENTS 

			if (k - Itrans_start)*dt > Itrans_duration :
				self.thalamus.fr = 0.
				if (not dmts and self.poisson.sample().item() > 1) or (dmts and k*dt in cue_timings):
					self.thalamus.fr = self.thalamus.Itrans
					Itrans_start = k
					stim.append(k*dt)
			
			if distractor and (k - distractor_start)*dt > distractor_duration:
				try : self.thalamus2.fr = 0
				except :
					print('Two columns are needed for using distractors. Aborting...')
					sys.exit()
				if k*dt == distractor_timing:
					self.thalamus2.fr = self.thalamus2.Itrans 
					distractor_start = k

			if nic_trans and (k - nic_trans_start)*dt > nic_trans_duration :
				try : self.cholinergic_nuclei.fr = 0
				except : 
					print('Cholinergic innervation is needed for this mode')
					sys.exit()
				if k*dt == nic_trans_timing:
					self.cholinergic_nuclei.fr = self.cholinergic_nuclei.Itrans 
					nic_trans_start = k

			# STEPS

			for pop in self.populations : pop.get_derivative(dt, self.n.sample().item())
			for pop in self.populations : pop.step(dfr_computed = True)
			
			for i in range(self.dim): tsr[i, k] = self.populations[i].fr
			tsr[-1, k] = k*dt

		for pop in self.populations : pop.reset()		
		
		self.simulations.append((Simulation(self, tsr, stim, sim_prm, reject, info, plot)))

		return self.simulations[-1]

		# Shouldn't fr be constrained to positive values earlier in the code ? 


	def equilibria(self): 

		"""
		
		Returns all possible dynamical equilibria of the deterministic model. 
		
		Entries: 
		-------
		
			info = bool : optional argument warning when tested parameter set leads to non-bistable dynamics.
		
		Outputs:
		-------       
		
			S = [[...], [...], ...] : NumPy-type list of 4-dimensional coordinates of dynamical equilibria.
		
		"""

		dt = 0.01

		def F(x):

			#[pop.fr for pop in self.populations] = x
			for i in range(self.dim) : self.populations[i].fr = x[i]
			derivatives = [pop.get_derivative(dt) for pop in self.populations]
			return derivatives
		
		for pop in self.populations : pop.deterministic()

		S = [np.random.uniform(size=self.dim)]
		for k in range(1000):
			x0 = [400*np.random.uniform(size=self.dim) - 200]
			sol = optimize.root(F, x0, method='hybr')
			#sol = optimize.fsolve(F, x0)
			if not np.isclose(sol.x, S, atol=self.atol, rtol=self.rtol).any():
				if np.isclose(F(sol.x), [0. for i in range(self.dim)], atol=self.atol, rtol=self.rtol).all():
					S.append(sol.x)

		for pop in self.populations : pop.reset()

		S = S[1:]

		# ATTEMPTING TO CUT SATURATED EQUILIBRIA OFF, gotta go back to this
		'''
		for s in S:
			if not np.array([fr < .45 for fr in [s[0] / self.Ae, s[1] / self.Ap, s[2] / self.As, s[3] /self.Av]]).all() :
				# S.remove(s)	HERE bug : the truth value of an array with more than one element is ambiguous
				self.hidden_eq = True
		'''

		return np.sort(S, 0)


	def free(): # Free memory from this column and every attached population
		pass 


	def grow_single_column(self, mod_prm, dof):

		### FIXED MODEL PARAMETERS

		# /!\ Tref shouldn't be the same for all neurons ! 
		
		tau_m, tau_m_PVs, tau_adp = mod_prm[0].item(), mod_prm[1].item(), mod_prm[2].item()	 # Circuit and adaptation time constants [s](float) ; Amount of divisive inhibition [1](float).
		Itrans = mod_prm[3].item()
		gaba = 1. + mod_prm[4].item() # Conductance of GABAergic projections [1](float).
		a_e, a_p, a_s, a_v = mod_prm[5].item(), mod_prm[6].item(), mod_prm[7].item(), mod_prm[8].item() # Maximal slopes of populational response functions [1](float).
		b_e, b_p, b_s, b_v = mod_prm[9].item(), mod_prm[10].item(), mod_prm[11].item(), mod_prm[12].item() # Critical thresholds of populational response functions [1](float). 
		Tref_PYR, Tref_INs = mod_prm[14], mod_prm[15]	
		
	### FREE MODEL PARAMETERS (DEGREES OF FREEDOM)
		
		# Scaling factors [spikes/min](float).
		Ae, Ap, As, Av = dof[0].item(), dof[1].item(), dof[2].item(), dof[3].item()

		# Normalized synaptic weights [1 -> min/spikes](float).
		wee, wpe, wse = dof[4].item() / Ae, gaba * dof[5].item() / Ap, gaba * dof[6].item() / As 		## MAYBE we should normalize the weights in the Synapse() class ? (has access to presynaptic amplitude)
		wes, wvs = dof[7].item() / Ae, dof[8].item() / Av 
		wep, wpp, wvp, wsp = dof[9].item() / Ae, gaba * dof[10].item() / Ap, .3*wvs, gaba * dof[11].item() / As
		wev, wsv = dof[12].item() / Ae, gaba * dof[13].item() / As

		# External currents [1](float).
		Ie_ext, Ip_ext, Is_ext, Iv_ext = dof[14].item(), dof[15].item(), dof[16].item(), dof[17].item()	

		# Bistable dynamics parameters.
		q, J_adp, sigma = dof[18].item(), dof[19].item() / Ae, dof[20].item()

			# /!\ WARNING # 
		# Originally dof[21] is usf. We decide to make it fixed (included as mod_prm[12]) and shorten dof to length of 21. 
		# Also, we fix I_trans (originally dof[18] as mod_prm[13]. Instead, 
		# we define q as a free parameter (originally defined above as mod[2])

	### POPULATIONS & CONNEXIONS INSTANTIATION

		pyr = NeuralPop('pyr', self, Ae, sigma, tau_m, tau_adp, a_e, b_e, Ie_ext, NT = 'glutamate', Tref = Tref_PYR, J_adp = J_adp)
		pv  = NeuralPop('pv',  self, Ap, sigma, tau_m_PVs, tau_adp, a_p, b_p, Ip_ext, NT = 'gaba', Tref = Tref_INs)
		som = NeuralPop('som', self, As, sigma, tau_m, tau_adp, a_s, b_s, Is_ext, NT = 'gaba', Tref = Tref_INs, J_adp = 0)
		vip = NeuralPop('vip', self, Av, sigma, tau_m, tau_adp, a_v, b_v, Iv_ext, NT = 'gaba', Tref = Tref_INs, J_adp = 0)
		thalamus = NeuralPop('thalamus', self, None, None, None, None, None, None, None, NT = 'glutamate', Itrans = Itrans)

		pyr.synapse(pyr, wee)
		if self.STP : pyr.synapse(pv, wpe, q, STP = 'd', stained = False)
		else : pyr.synapse(pv, wpe, q)
		pyr.synapse(som, wse)
		
		if self.STP : som.synapse(pyr, wes, STP = 'f', stained = False)
		else : som.synapse(pyr, wes) 
		som.synapse(vip, wvs)

		if self.STP : pv.synapse(pyr, wep, STP = 'd', stained = False)
		else : pv.synapse(pyr, wep)
		pv.synapse(pv, wpp)
		pv.synapse(vip, wvp)
		pv.synapse(som, wsp)
		vip.synapse(pyr, wev)
		vip.synapse(som, wsv)

		wbu = 0.01 # bottum-up weight

		pyr.synapse(thalamus, 1*wbu)
		if self.STP : pv.synapse(thalamus, 0.5*wbu, STP = 'd')
		else : pv.synapse(thalamus, 0.5*wbu) 
		vip.synapse(thalamus, 0.5*wbu)
		
		#self.thalamus2 = NeuralPop('thalamus', None, None, None, None, None, None, None, NT = 'glutamate', Itrans = Itrans)
		#pv.synapse(thalamus, 1)

		populations = [pyr, pv, som, vip]

		return thalamus, populations


	def grow_two_columns(self, mod_prm, dof, pdr = 1, distractor_amplitude = 3, nic_trans_amplitude = 0.5):  	# pdr : Proximal / Distal Ratio of synaptic strengths. We alse have to set distal wes and nicotinic transient amplitude so that they can be defined from main.

		### FIXED MODEL PARAMETERS

		# /!\ Tref shouldn't be the same for all neurons ! 
		
		tau_m, tau_m_PVs, tau_adp = mod_prm[0].item(), mod_prm[1].item(), mod_prm[2].item()	 # Circuit and adaptation time constants [s](float) ; Amount of divisive inhibition [1](float).
		Itrans = mod_prm[3].item()
		gaba = 1. + mod_prm[4].item() # Conductance of GABAergic projections [1](float).
		a_e, a_p, a_s, a_v = mod_prm[5].item(), mod_prm[6].item(), mod_prm[7].item(), mod_prm[8].item() # Maximal slopes of populational response functions [1](float).
		b_e, b_p, b_s, b_v = mod_prm[9].item(), mod_prm[10].item(), mod_prm[11].item(), mod_prm[12].item() # Critical thresholds of populational response functions [1](float).
		Tref_PYR, Tref_INs = mod_prm[14], mod_prm[15]	
		
	### FREE MODEL PARAMETERS (DEGREES OF FREEDOM)
		
		# Scaling factors [spikes/min](float).
		Ae, Ap, As, Av = dof[0].item(), dof[1].item(), dof[2].item(), dof[3].item()

		# Normalized synaptic weights [1 -> min/spikes](float).
		wee, wpe, wse = dof[4].item() / Ae, gaba * dof[5].item() / Ap, gaba * dof[6].item() / As 		## MAYBE we should normalize the weights in the Synapse() class ? (has access to presynaptic amplitude)
		wes, wvs = dof[7].item() / Ae, dof[8].item() / Av 
		wep, wpp, wvp, wsp = dof[9].item() / Ae, gaba * dof[10].item() / Ap, .3*wvs, gaba * dof[11].item() / As
		wev, wsv = dof[12].item() / Ae, gaba * dof[13].item() / As

		# External currents [1](float).
		Ie_ext, Ip_ext, Is_ext, Iv_ext = dof[14].item(), dof[15].item(), dof[16].item(), dof[17].item()	

		# Adaptation of external currents for two populations : 
		re, rs, rv = 1.1926, 35, 1.0847 # We use UP-state rs because activated by the other pop 
		re, rs, rv = 1.4090, 1.2391, 1.6413
		Is_ext -= pdr*wes*re
		#Ip_ext = Ip_ext + wvp*rv + wsp*35 - wep*re
		Ip_ext = Ip_ext + wvp*rv - wep*re + wsp*32

		# Bistable dynamics parameters.
		q, J_adp, sigma = dof[18].item(), dof[19].item() / Ae, dof[20].item()

		beta2 = 1.5 # /!\ We consider 0.5 to be muscarinic, never make it drop to 0 ! 
		alpha5 = 0.5
		alpha7 = 0.25

		Itrans_ACh = nic_trans_amplitude

			# /!\ WARNING # 
		# Originally dof[21] is usf. We decide to make it fixed (included as mod_prm[12]) and shorten dof to length of 21. 
		# Also, we fix I_trans (originally dof[18] as mod_prm[13]. Instead, 
		# we define q as a free parameter (originally defined above as mod[2])

	### POPULATIONS & CONNEXIONS INSTANTIATION

		pyr1 = NeuralPop('pyr', self, Ae, sigma, tau_m, tau_adp, a_e, b_e, Ie_ext, NT = 'glutamate', Tref = Tref_PYR, J_adp = J_adp)
		pyr2 = NeuralPop('pyr', self, Ae, sigma, tau_m, tau_adp, a_e, b_e, Ie_ext, NT = 'glutamate', Tref = Tref_PYR, J_adp = J_adp)
		pv  = NeuralPop('pv',  self, Ap, sigma, tau_m_PVs, tau_adp, a_p, b_p, Ip_ext, NT = 'gaba', Tref = Tref_INs)
		som1 = NeuralPop('som', self, As, sigma, tau_m, tau_adp, a_s, b_s, Is_ext, NT = 'gaba', Tref = Tref_INs, J_adp = 0)
		som2 = NeuralPop('som', self, As, sigma, tau_m, tau_adp, a_s, b_s, Is_ext, NT = 'gaba', Tref = Tref_INs, J_adp = 0)
		vip1 = NeuralPop('vip', self, Av, sigma, tau_m, tau_adp, a_v, b_v, Iv_ext, NT = 'gaba', Tref = Tref_INs, J_adp = 0)
		vip2 = NeuralPop('vip', self, Av, sigma, tau_m, tau_adp, a_v, b_v, Iv_ext, NT = 'gaba', Tref = Tref_INs, J_adp = 0)

		thalamus1 = NeuralPop('thalamus', self, None, None, None, None, None, None, None, NT = 'glutamate', Itrans = Itrans)
		thalamus2 = NeuralPop('thalamus', self, None, None, None, None, None, None, None, NT = 'glutamate', Itrans = distractor_amplitude)
		cholinergic_nuclei = NeuralPop('cholinergic_nuclei', self, None, None, None, None, None, None, None, NT = 'ACh', Itrans = Itrans_ACh)

		# INTRACOLUMNAR SYNAPSES # 

		# Column 1

		pyr1.synapse(pyr1, wee)
		if self.STP : pyr1.synapse(pv, wpe, q, STP = 'd', stained = False)
		else : pyr1.synapse(pv, wpe, q)
		pyr1.synapse(som1, wse)

		if self.STP : som1.synapse(pyr1, wes, STP = 'f', stained = False)
		else : som1.synapse(pyr1, wes)
		som1.synapse(vip1, wvs)

		if self.STP : pv.synapse(pyr1, wep, STP = 'd')
		else : pv.synapse(pyr1, wep)
		pv.synapse(pv, wpp)
		pv.synapse(vip1, wvp)
		pv.synapse(som1, wsp)

		vip1.synapse(pyr1, wev)
		vip1.synapse(som1, wsv)

		wbu = 0.01

		pyr1.synapse(thalamus1, 1*wbu)
		if self.STP : pv.synapse(thalamus1, 0.5*wbu, STP = 'd', stained = False)
		else : pv.synapse(thalamus1, 0.5*wbu)
		vip1.synapse(thalamus1, 0.5*wbu)

		# Column 2

		pyr2.synapse(pyr2, wee)
		if self.STP : pyr2.synapse(pv, wpe, q, STP = 'd')
		else : pyr2.synapse(pv, wpe, q)
		pyr2.synapse(som2, wse)

		if self.STP : som2.synapse(pyr2, wes, STP = 'f')
		else : som2.synapse(pyr2, wes)
		som2.synapse(vip2, wvs)

		if self.STP : pv.synapse(pyr2, wep, STP = 'd')
		else : pv.synapse(pyr2, wep)
		pv.synapse(vip2, wvp)
		pv.synapse(som2, wsp)

		vip2.synapse(pyr2, wev)
		vip2.synapse(som2, wsv)

		pyr2.synapse(thalamus2, 1*wbu)
		#if self.STP : pv.synapse(thalamus2, 0.5*wbu, STP = 'd')
		#else : pv.synapse(thalamus2, 0.5*wbu)
		vip2.synapse(thalamus2, 0.5*wbu)

		# INTERCOLUMNAR SYNAPSES # 
		
		#pyr1.synapse(som2, pdr*wse)
		#pyr2.synapse(som1, pdr*wse)

		if self.STP : som1.synapse(pyr2, pdr*wes, STP = 'f') # Lets say for now PYR cells connect just as much to intra than intercolumnar SOMs
		else : som1.synapse(pyr2, wes)
		if self.STP : som2.synapse(pyr1, pdr*wes, STP = 'f')
		else : som2.synapse(pyr1, wes)
		
		# FOREBRAIN SYNAPSES #

		vip1.synapse(cholinergic_nuclei, 0.5*wbu, nic_normalization = self.nic_normalization)
		vip2.synapse(cholinergic_nuclei, 0.5*wbu, nic_normalization = self.nic_normalization)
		if not self.kamigaki : som1.synapse(cholinergic_nuclei, 2*wbu, STP = 'd', nic_normalization = self.nic_normalization, stained = True)
		if not self.kamigaki : som2.synapse(cholinergic_nuclei, 2*wbu, STP = 'd', nic_normalization = self.nic_normalization)
		#pv.synapse(cholinergic_nuclei, 0.5, STP = 'd')

		populations = [pyr1, pv, som1, vip1, pyr2, som2, vip2]

		return thalamus1, thalamus2, cholinergic_nuclei, populations
