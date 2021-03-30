import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
from NeuralPop import NeuralPop
from Simulation import Simulation

# An "instance" of CCM stands for one cortical column with defined parameters. 
# Once declared at the instanciation, the latter only change if explicitly modified (functions ...(), ...())
# Conversely, simulation parameters are easily redeclared for each simulation. 


class CCM():

	def __init__(self, dof, mod_prm, sim_prm = None, equilibria = False, reject = False, computer = "cpu", enc = torch.float64):

		# TEMPORARY, find a way to better deal with this
		self.dev = torch.device(computer)

		self.sim_prm = sim_prm
		self.mod_prm = mod_prm
		self.dof = dof

		self.reject = reject

		self.simulations = []

		### FIXED MODEL PARAMETERS

		# /!\ Tref shouldn't be the same for all neurons ! 
		
		tau_m, tau_adp = mod_prm[0].item(), mod_prm[1].item()	 # Circuit and adaptation time constants [s](float) ; Amount of divisive inhibition [1](float).
		self.I_trans = mod_prm[2].item()
		gaba = 1. + mod_prm[3].item() # Conductance of GABAergic projections [1](float).
		a_e, a_p, a_s, a_v = mod_prm[4].item(), mod_prm[5].item(), mod_prm[6].item(), mod_prm[7].item() # Maximal slopes of populational response functions [1](float).
		b_e, b_p, b_s, b_v = mod_prm[8].item(), mod_prm[9].item(), mod_prm[10].item(), mod_prm[11].item() # Critical thresholds of populational response functions [1](float).
		self.usf = mod_prm[12] # Frequency of ultra-slow fluctuations 
		Tref = mod_prm[13]

		# External currents [1](float).
		Ie_ext, Ip_ext, Is_ext, Iv_ext = dof[14].item(), dof[15].item(), dof[16].item(), dof[17].item()		
		
	### FREE MODEL PARAMETERS (DEGREES OF FREEDOM)
		
		# Scaling factors [spikes/min](float).
		Ae, Ap, As, Av = dof[0].item(), dof[1].item(), dof[2].item(), dof[3].item()

		# Normalized synaptic weights [1 -> min/spikes](float).
		wee, wpe, wse = dof[4].item() / Ae, gaba * dof[5].item() / Ap, gaba * dof[6].item() / As 		## MAYBE we should normalize the weights in the Synapse() class ? (has access to presynaptic amplitude)
		wes, wvs = dof[7].item() / Ae, dof[8].item() / Av 
		wep, wpp, wvp, wsp = dof[9].item() / Ae, gaba * dof[10].item() / Ap, .5*wvs, gaba * dof[11].item() / As
		wev, wsv = dof[12].item() / Ae, gaba * dof[13].item() / As

		# Bistable dynamics parameters.
		q, J_adp, sigma = dof[18].item(), dof[19].item() / Ae, dof[20].item()

			# /!\ WARNING # 
		# Originally dof[21] is usf. We decide to make it fixed (included as mod_prm[12]) and shorten dof to length of 21. 
		# Also, we fix I_trans (originally dof[18] as mod_prm[13]. Instead, 
		# we define q as a free parameter (originally defined above as mod[2])

	### POPULATIONS & CONNEXIONS INSTANCIATION

		self.pyr = NeuralPop('pyr', Ae, sigma, tau_m, tau_adp, a_e, b_e, Ie_ext, NT = 'glutamate', Tref = Tref, J_adp = J_adp)
		self.pv  = NeuralPop('pv',  Ap, sigma, tau_m, tau_adp, a_p, b_p, Ip_ext, NT = 'gaba', Tref = Tref)
		self.som = NeuralPop('som', As, sigma, tau_m, tau_adp, a_s, b_s, Is_ext, NT = 'gaba', Tref = Tref)
		self.vip = NeuralPop('vip', Av, sigma, tau_m, tau_adp, a_v, b_v, Iv_ext, NT = 'gaba', Tref = Tref)

		self.pyr.synapse(self.pyr, wee)
		self.pyr.synapse(self.pv, wpe, q)
		self.pyr.synapse(self.som, wse)

		self.som.synapse(self.pyr, wes)
		self.som.synapse(self.vip, wvs)

		self.pv.synapse(self.pyr, wep)
		self.pv.synapse(self.pv, wpp)
		self.pv.synapse(self.vip, wvp)
		self.pv.synapse(self.som, wsp)

		self.vip.synapse(self.pyr, wev)
		self.vip.synapse(self.som, wsv)

		self.populations = [self.pyr, self.pv, self.som, self.vip]


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
			self.critic = self.S[1]
		else : 
			self.S = [None]
			self.critic = torch.as_tensor([10,10,10,10], device = self.dev, dtype = torch.float64)


	def simulate(self, sim_prm = None, info=False, plot=False, dmts = False, cue_timings = [1], reject = None): # Maybe put reject in sim_prm
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
		if reject and len(self.S < 3):
			if info : print('Model not bistable')
			return Simulation(None, None, sim_prm, self.mod_prm, self.dof, self.critic, aborted = True)

		
		window = sim_prm[0].item() # Stimulation window [s](float).
		dt = sim_prm[1].item() # Time resolution [s](float).
		atol, rtol = sim_prm[2].item(), sim_prm[3].item() # Absolute and relative tolerances for float comparison.
		
		if torch.isnan(sim_prm[4]):
			torch.manual_seed(time.time())
		else:
			torch.manual_seed(sim_prm[4])
		
		#smin = round(3 * max(self.tau, self.tau_adp) / dt) # Starting time for stimulation window [1](int).
		smin = 10 			# CORRECT THAT
		smax = round(window / dt) # End of stimulation window [1](int).
		N = smin + smax # Total number of time-points [1](int)


	### SIMULATION 

		if plot:
			tsr = torch.empty((19, N), device=self.dev, dtype=torch.float64)
		else:
			tsr = torch.empty((9, N), device=self.dev, dtype=torch.float64)
		stim = []
	
		for k in range(N):
			
			trans = 0.
			if not dmts:
				if self.poisson.sample().item() > 1: # Stimulation trigger.
					trans = self.I_trans
					stim.append(k*dt)
			else : 
				if k*dt in cue_timings:
					trans = self.I_trans

			self.pyr.step(dt, self.n.sample().item(), trans = trans)
			self.pv.step(dt, self.n.sample().item())
			self.som.step(dt, self.n.sample().item())
			self.vip.step(dt, self.n.sample().item())
			
			t = k * dt # Time [s](float).
			tsr[self.TT, k] = t
			
			if plot:  # FIX THAT AS WELL
				tsr[self.RE, k], tsr[self.dRE, k], tsr[self.INE_E, k], tsr[self.INS_E, k], tsr[self.IND_E, k], tsr[self.IE_ADP, k] = max(re, 0.), dre, ine_e, ins_e, ind_e, Ie_adp
				tsr[self.RP, k], tsr[self.dRP, k], tsr[self.INE_P, k], tsr[self.INS_P, k] = max(rp, 0.), drp, ine_p, ins_p
				tsr[self.RS, k], tsr[self.dRS, k], tsr[self.INE_S, k], tsr[self.INS_S, k] = max(rs, 0.), drs, ine_s, ins_s
				tsr[self.RV, k], tsr[self.dRV, k], tsr[self.INE_V, k], tsr[self.INS_V, k] = max(rv, 0.), drv, ine_v, ins_v
			else:
				tsr[self.RE, k], tsr[self.RP, k], tsr[self.RS, k], tsr[self.RV, k] = self.pyr.fr, self.pv.fr, self.som.fr, self.vip.fr


		for pop in self.populations : pop.reset()		
		
		self.simulations.append((Simulation(tsr, stim, sim_prm, self.mod_prm, self.dof, reject, info, plot, self.critic)))

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
		def F(x):

			dt = 0.01

			for pop in self.populations : pop.deterministic()

			self.pyr.fr = x[0]
			self.pv.fr = x[1]
			self.som.fr = x[2]
			self.vip.fr = x[3]

			dre = self.pyr.get_derivative(dt)
			drp = self.pv.get_derivative(dt)
			drs = self.som.get_derivative(dt)
			drv = self.vip.get_derivative(dt)

			for pop in self.populations : pop.reset()

			return [dre, drp, drs, drv]
			
			#self.pyr.fr, self.pv.fr, self.som.fr, self.vip.fr = x[0], x[1], x[2], x[3]
			#return [self.pyr.get_derivative(self.dt), self.pv.get_derivative(self.dt), self.som.get_derivative(self.dt), self.pv.get_derivative(self.dt)]

		self.atol, self.rtol = 1e-12, 1e-3
		
		S = [np.random.uniform(size=4)]
		for k in range(1000):
			x0 = [400*np.random.uniform(size=4) - 200]
			sol = optimize.root(F, x0, method='hybr')
			#sol = optimize.fsolve(F, x0)
			if not np.isclose(sol.x, S, atol=self.atol, rtol=self.rtol).any():
				if np.isclose(F(sol.x), [0., 0., 0., 0.], atol=self.atol, rtol=self.rtol).all():
					S.append(sol.x)

		S = S[1:]

		# ATTEMPTING TO CUT SATURATED EQUILIBRIA OFF, gotta go back to this
		'''
		for s in S:
			if not np.array([fr < .45 for fr in [s[0] / self.Ae, s[1] / self.Ap, s[2] / self.As, s[3] /self.Av]]).all() :
				# S.remove(s)	HERE bug : the truth value of an array with more than one element is ambiguous
				print('smt')
				self.hidden_eq = True
		'''

		return np.sort(S, 0)


	def free(): # Free memory from this column and every attached population
		pass 
