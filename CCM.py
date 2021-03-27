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

	def __init__(self, dof, mod_prm, sim_prm = None, equilibria = False, computer = "cpu", enc = torch.float64):

		# TEMPORARY, find a way to better deal with this
		self.dev = torch.device(computer)

		self.sim_prm = sim_prm
		self.mod_prm = mod_prm
		self.dof = dof

		self.simulations = []

		### FIXED MODEL PARAMETERS
		
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

		## GOTTA ADD THE "SELF" 

		self.pyr = NeuralPop('pyr', Ae, sigma, tau_m, tau_adp, a_e, b_e, Ie_ext, NT = 'glutamate', Tref = Tref, J_adp = J_adp)
		self.pv  = NeuralPop('pv',  Ap, sigma, tau_m, tau_adp, a_p, b_p, Ip_ext, NT = 'gaba', Tref = Tref)
		self.som = NeuralPop('som', As, sigma, tau_m, tau_adp, a_s, b_s, Is_ext, NT = 'gaba', Tref = Tref)
		self.vip = NeuralPop('vip', Av, sigma, tau_m, tau_adp, a_v, b_v, Iv_ext, NT = 'gaba', Tref = Tref)

		self.pyr.synapse(self.pyr, wee, printt = True)
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
		else : self.S = [None]


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

	### SIMULATION PARAMETERS
		if sim_prm == None : sim_prm = self.sim_prm
		if sim_prm == None : 
			print('No simulation parameters declared')
			return 
		
		window = sim_prm[0].item() # Stimulation window [s](float).
		dt = sim_prm[1].item() # Time resolution [s](float).
		atol, rtol = sim_prm[3].item(), sim_prm[4].item() # Absolute and relative tolerances for float comparison.
		
		if torch.isnan(sim_prm[5]):
			torch.manual_seed(time.time())
		else:
			torch.manual_seed(sim_prm[5])
		
		#smin = round(3 * max(self.tau, self.tau_adp) / dt) # Starting time for stimulation window [1](int).
		smin = 10 			# CORRECT THAT
		smax = round(window / dt) # End of stimulation window [1](int).
		N = smin + smax # Total number of time-points [1](int)


	### SIMULATION 

		re, dre, Ie_adp, ine, rs, drs, rp, drp, rv, drv, t = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
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


		print('\nSYNAPSES :')
		for pop in [self.pyr, self.pv, self.som, self.vip]:
			print('NAME :\t', pop.name)
			print('EXCITATORY :\t', [syn.presyn.name for syn in pop.glut_synapses])
			print('INHIBITORY :\t', [syn.presyn.name for syn in pop.gaba_synapses])


		self.pyr.reset()
		self.pv.reset()
		self.som.reset()
		self.vip.reset()		
		
		self.simulations.append((Simulation(tsr, stim, sim_prm, self.mod_prm, self.dof, self.S, reject, info, plot)))

		return self.simulations[-1]

		# Shouldn't fr be constrained to positive values earlier in the code ? 


	def equilibria(self):
		## TRY and modify atol & rtol or smt to get rid of the 4 equilibria when there're only 3... 

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
			self.tr = 1. 		# WTF ?!
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
