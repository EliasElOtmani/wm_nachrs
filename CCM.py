import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
from Simulation import Simulation

# An "instance" of CCM stands for one cortical column with defined parameters. 
# Once declared at the instanciation, the latter only change if explicitly modified (functions ...(), ...())
# Conversely, simulation parameters are easily redeclared for each simulation. 

class CCM():

	def __init__(self, dof, mod_prm, sim_prm = None, refractory_period = True, equilibria = False, computer = "cpu", enc = torch.float64):

		# TEMPORARY, find a way to better deal with this
		self.enc = enc
		self.dev = torch.device(computer)

		self.sim_prm = sim_prm
		self.mod_prm = mod_prm
		self.dof = dof

		self.simulations = []

	### FIXED MODEL PARAMETERS
		
		self.tau, self.tau_adp = mod_prm[0].item(), mod_prm[1].item()	 # Circuit and adaptation time constants [s](float) ; Amount of divisive inhibition [1](float).
		self.I_trans = mod_prm[2].item()
		self.gaba = 1. + mod_prm[3].item() # Conductance of GABAergic projections [1](float).
		self.a_e, self.a_p, self.a_s, self.a_v = mod_prm[4].item(), mod_prm[5].item(), mod_prm[6].item(), mod_prm[7].item() # Maximal slopes of populational response functions [1](float).
		self.b_e, self.b_p, self.b_s, self.b_v = mod_prm[8].item(), mod_prm[9].item(), mod_prm[10].item(), mod_prm[11].item() # Critical thresholds of populational response functions [1](float).
		self.usf = mod_prm[12] # Frequency of ultra-slow fluctuations 
		self.tr = mod_prm[13].item() # Refractory time [in unit 'dt'](float).
		
		
	### FREE MODEL PARAMETERS (DEGREES OF FREEDOM)
		
		# Scaling factors [spikes/min](float).
		self.Ae, self.Ap, self.As, self.Av = dof[0].item(), dof[1].item(), dof[2].item(), dof[3].item()


		# Normalized synaptic weights [1 -> min/spikes](float).
		self.wee, self.wpe, self.wse = dof[4].item() / self.Ae, self.gaba * dof[5].item() / self.Ap, self.gaba * dof[6].item() / self.As 
		self.wes, self.wvs = dof[7].item() / self.Ae, dof[8].item() / self.Av 
		self.wep, self.wpp, self.wvp, self.wsp = dof[9].item() / self.Ae, self.gaba * dof[10].item() / self.Ap, .5*self.wvs, self.gaba * dof[11].item() / self.As
		self.wev, self.wsv = dof[12].item() / self.Ae, self.gaba * dof[13].item() / self.As

		
		# External currents [1](float).
		self.Ie_ext, self.Ip_ext, self.Is_ext, self.Iv_ext = dof[14].item(), dof[15].item(), dof[16].item(), dof[17].item()
		
		# Bistable dynamics parameters.
		self.q, self.J_adp, self.sigma = dof[18].item(), dof[19].item() / self.Ae, dof[20].item()

		# /!\ WARNING # 
		# Originally dof[21] is usf. We decide to make it fixed (included as mod_prm[12]) and shorten dof to length of 21. 
		# Also, we fix I_trans (originally dof[18] as mod_prm[13]. Instead, 
		# we define q as a free parameter (originally defined above as mod[2])


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
		if equilibria : self.S = self.equilibria()
		else : self.S = None


	# Populational response functions, as derived from [Papasavvas, 2015].   MAYBE DEFINE THEM IN THE __INIT__() ? OR IN THE SIMULATION (easier if we want to be able to create the simpler_CCM from the same class) ?  
	def f(self, a_, b_, x, b, a):
		return .5 * ( 1. + torch.tanh( torch.as_tensor([.5 * ( (a_ / (1. + a)) * (x - b - b_) )]) ).item() ) - ( 1./(1. + torch.exp( torch.as_tensor([a_*b_/(1.+a)]) ).item() ) )
	
	# Asymptotic plateau of populational response functions, as derived from [Papasavvas, 2015]
	def K(self, a_, b_, a):
		return torch.exp(torch.as_tensor([a_*b_/(1.+a)])).item() / (1. + torch.exp(torch.as_tensor([a_*b_/(1.+a)])).item())

	def simulate(self, sim_prm = None, reject=True, info=False, plot=False, dmts = False, cue_timings = [1]):
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

		if self.reject and equilibria and len(self.S)<3:
			if self.info : print("Model not bistable")
			traces = None
			return Simulation(traces, stimuli, sim_prm, mod_prm, dof, S, reject, info, plot, empty = True)

	### SIMULATION PARAMETERS
		if sim_prm == None : sim_prm = self.sim_prm
		if sim_prm == None : 
			print('No simulation parameters declared')
			return 
		
		window = sim_prm[0].item() # Stimulation window [s](float).
		dt = sim_prm[1].item() # Time resolution [s](float).
		atol, rtol = sim_prm[2].item(), sim_prm[3].item() # Absolute and relative tolerances for float comparison.
		
		if torch.isnan(sim_prm[4]):
			torch.manual_seed(time.time())
		else:
			torch.manual_seed(sim_prm[4])
		
		smin = round(3 * max(self.tau, self.tau_adp) / dt) # Starting time for stimulation window [1](int).
		smax = round(window / dt) # End of stimulation window [1](int).
		N = smin + smax # Total number of time-points [1](int)


	### SIMULATION 

		re, dre, Ie_adp, ine, rs, drs, rp, drp, rv, drv, t = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
		if plot:
			tsr = torch.empty((19, N), device=self.dev, dtype=torch.float64)
		else:
			tsr = torch.empty((9, N), device=self.dev, dtype=torch.float64)
		#s = self.poisson.sample().item() + smin 	# WHAT'S THIS USEFUL FOR ? No effect of removing it...
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
				
			# PYR activity.
			Ie = self.Ie_ext + trans # External input [1](float).
			ine_e = self.wee*re - Ie_adp + Ie # Excitatory input [1](float).
			ins_e = self.wse*rs + (1-self.q)*self.wpe*rp # Substractive inhibitory input [1](float). 
			ind_e = self.q*self.wpe*rp # Divisive inhibitory input [1](float). 
			Ie_adp += (dt/self.tau_adp) * ( - Ie_adp + re * self.J_adp ) # PYR activity-dependent adaptation current [1](float).
			Ke = self.K(self.a_e, self.b_e, ind_e) # Response function plateau [1](float).
			dre = (dt/self.tau) * ( - re + (self.Ae*Ke - self.tr*re)*self.f(self.a_e, self.b_e, ine_e, ins_e, ind_e)  ) + self.Ae * Ke * np.sqrt(dt) * self.sigma * self.n.sample().item() # PYR activity derivative [spikes/min²](float).
			re += dre # PYR activity rate [spikes/min](float).
			
			# PV activity.
			ine_p = self.wep*re + self.Ip_ext # Excitatory input [1](float).
			ins_p = self.wpp*rp + self.wvp*rv + self.wsp*rs # Substractive inhibitory input [1](float).
			ind_p = 0. # Divisive inhibitory input [1](float).
			Kp = self.K(self.a_p, self.b_p, ind_p) # Response function plateau [1](float).
			drp = (self.dt/self.tau) * ( - rp + (self.Ap*Kp - self.tr*rp)*self.f(self.a_p, self.b_p, ine_p, ins_p, ind_p)  ) + self.Ap * Kp * np.sqrt(dt) * self.sigma * self.n.sample().item() # PV activity derivative [spikes/min²](float).
			rp += drp # PV activity rate [spikes/min](float).
			
			# SOM activity.
			ine_s = self.wes*re + self.Is_ext # Excitatory input [1](float).
			ins_s = self.wvs*rv # Substractive inhibitory input [1](float).
			ind_s = 0. # Divisive inhibitory input [1](float).
			Ks = self.K(self.a_s, self.b_s, ind_s) # Response function plateau [1](float).
			drs = (dt/self.tau) * ( - rs + (self.As*Ks - self.tr*rs)*self.f(self.a_s, self.b_s, ine_s, ins_s, ind_s)  ) + self.As * Ks * np.sqrt(dt) * self.sigma * self.n.sample().item() # SOM activity derivative [spikes/min²](float).
			rs += drs # SOM activity rate [spikes/min](float).

			# VIP activity.
			ine_v = self.wev*re + self.Iv_ext # Excitatory input [1](float).
			ins_v = self.wsv*rs # Substractive inhibitory input [1](float).
			ind_v = 0. # Divisive inhibitory input [1](float). 
			Kv = self.K(self.a_v, self.b_v, ind_v) # Response function plateau [1](float).
			drv = (dt/self.tau) * ( - rv + (self.Av*Kv - self.tr*rv)*self.f(self.a_v, self.b_v, ine_v, ins_v, ind_v)  ) + self.Av * Kv * np.sqrt(dt) * self.sigma * self.n.sample().item() # VIP activity derivative [spikes/min²](float).
			rv += drv # VIP activity rate [spikes/min](float).
			
			t = k * dt # Time [s](float).
			tsr[self.TT, k] = t

			re, rp, rs, rv = max(re, 0.), max(rp, 0.), max(rs, 0.), max(rv, 0.)
			
			if plot:
				tsr[self.RE, k], tsr[self.dRE, k], tsr[self.INE_E, k], tsr[self.INS_E, k], tsr[self.IND_E, k], tsr[self.IE_ADP, k] = max(re, 0.), dre, ine_e, ins_e, ind_e, Ie_adp
				tsr[self.RP, k], tsr[self.dRP, k], tsr[self.INE_P, k], tsr[self.INS_P, k] = max(rp, 0.), drp, ine_p, ins_p
				tsr[self.RS, k], tsr[self.dRS, k], tsr[self.INE_S, k], tsr[self.INS_S, k] = max(rs, 0.), drs, ine_s, ins_s
				tsr[self.RV, k], tsr[self.dRV, k], tsr[self.INE_V, k], tsr[self.INS_V, k] = max(rv, 0.), drv, ine_v, ins_v
			else:
				tsr[self.RE, k], tsr[self.RP, k], tsr[self.RS, k], tsr[self.RV, k] = max(re, 0.), max(rp, 0.), max(rs, 0.), max(rv, 0.)
		
		
		self.simulations.append((Simulation(tsr, stim, sim_prm, self.mod_prm, self.dof, self.S, reject, info, plot)))

		return self.simulations[-1]

		# Shouldn't fr be constrained to positive values earlier in the code ? 


	def equilibria(self):
		## TRY and modify atol & rtol or smt to get rid of the 4 equilibria when there're only 3... 
		# UPDATE : the temporary model we're using actually has 4 equilibria. They are all systematically found 
		# when increasing the number of random [re, rp, rv, rs] samples at the bottom of the function...

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

			# TEMPORARY - Maybe put the sim_prm as argument of equilibria ?..
			# But something's odd, equilibria shouldn't depend on simulation parameters
			self.dt = .01
			self.tr = 1.
			self.atol, self.rtol = 1e-12, 1e-3
	
			re, rp, rs, rv = x[0], x[1], x[2], x[3]
	
			Ie_adp = 0.
			Ie = self.Ie_ext - Ie_adp
			ine = self.wee*re + Ie
			ins = self.wse*rs + (1-self.q)*self.wpe*rp
			ind = self.q*self.wpe*rp
			Ke = self.K(self.a_e, self.b_e, ind)
			dre = (self.dt/self.tau)*( - re + (self.Ae*Ke - self.tr*re)*self.f(self.a_e, self.b_e, ine, ins, ind)  )
	
			ine = self.wep*re + self.Ip_ext
			ins = self.wpp*rp + self.wvp*rv + self.wsp*rs
			ind = 0.
			Kp = self.K(self.a_p, self.b_p, ind)
			drp = (self.dt/self.tau)*( - rp + (self.Ap*Kp - self.tr*rp)*self.f(self.a_p, self.b_p, ine, ins, ind)  )
		
			Is_adp = 0.
			ine = self.wes*re + self.Is_ext - Is_adp
			ins = self.wvs*rv
			ind = 0.
			Ks = self.K(self.a_s, self.b_s, ind)
			drs = (self.dt/self.tau)*( - rs + (self.As*Ks - self.tr*rs)*self.f(self.a_s, self.b_s, ine, ins, ind)  )

			ine = self.wev*re + self.Iv_ext
			ins = self.wsv*rs
			ind = 0.
			Kv = self.K(self.a_v, self.b_v, ind)
			drv = (self.dt/self.tau)*( - rv + (self.Av*Kv - self.tr*rv)*self.f(self.a_v, self.b_v, ine, ins, ind)  )
	
			return [dre, drp, drs, drv]
		
		S = [np.random.uniform(size=4)]
		for k in range(400):
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
