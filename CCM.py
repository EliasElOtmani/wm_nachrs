import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm

# Replace all the "if bool == True" with "if bool"

class CCM():

	def __init__(self, dof, mod_prm, sim_prm, reject=True, info=False, plot=False):

		# TEMPORARY, find a way to better deal with this
		computer = "cpu"
		self.enc = torch.float64
		self.dev = torch.device(computer)

	### FIXED MODEL PARAMETERS
		
		self.tau, self.tau_adp, self.q = mod_prm[0].item(), mod_prm[1].item(), mod_prm[2].item() # Circuit and adaptation time constants [s](float) ; Amount of divisive inhibition [1](float).
		self.gaba = 1. + mod_prm[3].item() # Conductance of GABAergic projections [1](float).
		self.a_e, self.a_p, self.a_s, self.a_v = mod_prm[4].item(), mod_prm[5].item(), mod_prm[6].item(), mod_prm[7].item() # Maximal slopes of populational response functions [1](float).
		self.b_e, self.b_p, self.b_s, self.b_v = mod_prm[8].item(), mod_prm[9].item(), mod_prm[10].item(), mod_prm[11].item() # Critical thresholds of populational response functions [1](float).
		
		
	### FREE MODEL PARAMETERS (DEGREES OF FREEDOM)
		
		# Scaling factors [spikes/min](float).
		self.Ae, self.Ap, self.As, self.Av = dof[0].item(), dof[1].item(), dof[2].item(), dof[3].item()
		
		# Normalized synaptic weights [1 -> min/spikes](float).
		self.wee, self.wpe, self.wse = dof[4].item() / self.Ae, self.gaba * dof[5].item() / self.Ae, self.gaba * dof[6].item() / self.Ae 
		self.wes, self.wvs = dof[7].item() / self.As, dof[8].item() / self.As 
		self.wep, self.wpp, self.wvp, self.wsp = dof[9].item() / self.Ap, self.gaba * dof[10].item() / self.Ap, .5*(self.Ap/self.As)*self.wvs, self.gaba * dof[11].item() / self.Ap
		self.wev, self.wsv = dof[12].item() / self.Av, self.gaba * dof[13].item() / self.Av
		
		# External currents [1](float).
		self.Ie_ext, self.Ip_ext, self.Is_ext, self.Iv_ext = dof[14].item(), dof[15].item(), dof[16].item(), dof[17].item()
		
		# Bistable dynamics parameters.
		self.I_trans, self.J_adp, self.sigma, self.usf = dof[18].item(), dof[19].item() / self.Ae, dof[20].item(), dof[21].item()
		
		
	### FIXED SIMULATION PARAMETERS
		
		self.window = sim_prm[0].item() # Stimulation window [s](float).
		self.dt = sim_prm[1].item() # Time resolution [s](float).
		self.tr = sim_prm[2].item() # Refractory time [in unit 'dt'](float).
		self.atol, self.rtol = sim_prm[3].item(), sim_prm[4].item() # Absolute and relative tolerances for float comparison.
		self.plot = plot
		self.info = info
		self.reject = reject
		
		if torch.isnan(sim_prm[5]) == True:
			torch.manual_seed(time.time())
		else:
			torch.manual_seed(sim_prm[5])
		
		self.smin = round(3 * max(self.tau, self.tau_adp) / self.dt) # Starting time for stimulation window [1](int).
		self.smax = round(self.window / self.dt) # End of stimulation window [1](int).
		self.N = self.smin + self.smax # Total number of time-points [1](int)
		
		# Short-hand indices for tensor-based data storage [1](int)
		self.RE, self.RP, self.RS, self.RV, self.dRE, self.dRP, self.dRS, self.dRV, self.TT = 0, 1, 2, 3, 4, 5, 6, 7, 8
		self.INE_E, self.INE_P, self.INE_S, self.INE_V, self.INS_E, self.INS_P, self.INS_S, self.INS_V = 9, 10, 11, 12, 13, 14, 15, 16
		self.IND_E, self.IE_ADP = 17, 18

	### MODEL
		
		# Sources of stochasticity.
		self.poisson = torch.distributions.poisson.Poisson(torch.tensor([self.usf])) # Poisson distribution of stimuli. 
		self.n = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])) # Normal distribution of neural noise.


	# Populational response functions, as derived from [Papasavvas, 2015].   MAYBE DEFINE THEM IN THE __INIT__() ? 
	def f(self, a_, b_, x, b, a):
		return .5 * ( 1. + torch.tanh( torch.as_tensor([.5 * ( (a_ / (1. + a)) * (x - b - b_) )]) ).item() ) - ( 1./(1. + torch.exp( torch.as_tensor([a_*b_/(1.+a)]) ).item() ) )
	
	# Asymptotic plateau of populational response functions, as derived from [Papasavvas, 2015]
	def K(self, a_, b_, a):
		return torch.exp(torch.as_tensor([a_*b_/(1.+a)])).item() / (1. + torch.exp(torch.as_tensor([a_*b_/(1.+a)])).item())

	def simulate(self):
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
		# Initialization
		re, dre, Ie_adp, ine, rs, drs, rp, drp, rv, drv, t = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
		if self.plot == True:
			tsr = torch.empty((19, self.N), device=self.dev, dtype=torch.float64)
		else:
			tsr = torch.empty((9, self.N), device=self.dev, dtype=torch.float64)
		s = self.poisson.sample().item() + self.smin
		stim = []
	
		for k in tqdm(range(self.N)):
			
			trans = 0.
			if self.poisson.sample().item() > 1: # Stimulation trigger.
				trans = self.I_trans
				stim.append(k*self.dt)
				
			# PYR activity.
			Ie = self.Ie_ext + trans # External input [1](float).
			ine_e = self.wee*re - Ie_adp + Ie # Excitatory input [1](float).
			ins_e = self.wse*rs + (1-self.q)*self.wpe*rp # Substractive inhibitory input [1](float). 
			ind_e = self.q*self.wpe*rp # Divisive inhibitory input [1](float). 
			Ie_adp += (self.dt/self.tau_adp) * ( - Ie_adp + re * self.J_adp ) # PYR activity-dependent adaptation current [1](float).
			Ke = self.K(self.a_e, self.b_e, ind_e) # Response function plateau [1](float).
			dre = (self.dt/self.tau) * ( - re + (self.Ae*Ke - self.tr*re)*self.f(self.a_e, self.b_e, ine_e, ins_e, ind_e)  ) + self.Ae * Ke * np.sqrt(self.dt) * self.sigma * self.n.sample().item() # PYR activity derivative [spikes/min²](float).
			re += dre # PYR activity rate [spikes/min](float).
			
			# PV activity.
			ine_p = self.wep*re + self.Ip_ext # Excitatory input [1](float).
			ins_p = self.wpp*rp + self.wvp*rv + self.wsp*rs # Substractive inhibitory input [1](float).
			ind_p = 0. # Divisive inhibitory input [1](float).
			Kp = self.K(self.a_p, self.b_p, ind_p) # Response function plateau [1](float).
			drp = (self.dt/self.tau) * ( - rp + (self.Ap*Kp - self.tr*rp)*self.f(self.a_p, self.b_p, ine_p, ins_p, ind_p)  ) + self.Ap * Kp * np.sqrt(self.dt) * self.sigma * self.n.sample().item() # PV activity derivative [spikes/min²](float).
			rp += drp # PV activity rate [spikes/min](float).
			
			# SOM activity.
			ine_s = self.wes*re + self.Is_ext # Excitatory input [1](float).
			ins_s = self.wvs*rv # Substractive inhibitory input [1](float).
			ind_s = 0. # Divisive inhibitory input [1](float).
			Ks = self.K(self.a_s, self.b_s, ind_s) # Response function plateau [1](float).
			drs = (self.dt/self.tau) * ( - rs + (self.As*Ks - self.tr*rs)*self.f(self.a_s, self.b_s, ine_s, ins_s, ind_s)  ) + self.As * Ks * np.sqrt(self.dt) * self.sigma * self.n.sample().item() # SOM activity derivative [spikes/min²](float).
			rs += drs # SOM activity rate [spikes/min](float).

			# VIP activity.
			ine_v = self.wev*re + self.Iv_ext # Excitatory input [1](float).
			ins_v = self.wsv*rs # Substractive inhibitory input [1](float).
			ind_v = 0. # Divisive inhibitory input [1](float). 
			Kv = self.K(self.a_v, self.b_v, ind_v) # Response function plateau [1](float).
			drv = (self.dt/self.tau) * ( - rv + (self.Av*Kv - self.tr*rv)*self.f(self.a_v, self.b_v, ine_v, ins_v, ind_v)  ) + self.Av * Kv * np.sqrt(self.dt) * self.sigma * self.n.sample().item() # VIP activity derivative [spikes/min²](float).
			rv += drv # VIP activity rate [spikes/min](float).
			
			t = k * self.dt # Time [s](float).
			tsr[self.TT, k] = t
			
			if self.plot == True:
				tsr[self.RE, k], tsr[self.dRE, k], tsr[self.INE_E, k], tsr[self.INS_E, k], tsr[self.IND_E, k], tsr[self.IE_ADP, k] = max(re, 0.), dre, ine_e, ins_e, ind_e, Ie_adp
				tsr[self.RP, k], tsr[self.dRP, k], tsr[self.INE_P, k], tsr[self.INS_P, k] = max(rp, 0.), drp, ine_p, ins_p
				tsr[self.RS, k], tsr[self.dRS, k], tsr[self.INE_S, k], tsr[self.INS_S, k] = max(rs, 0.), drs, ine_s, ins_s
				tsr[self.RV, k], tsr[self.dRV, k], tsr[self.INE_V, k], tsr[self.INS_V, k] = max(rv, 0.), drv, ine_v, ins_v
			else:
				tsr[self.RE, k], tsr[self.RP, k], tsr[self.RS, k], tsr[self.RV, k] = max(re, 0.), max(rp, 0.), max(rs, 0.), max(rv, 0.)
		
		return tsr, stim


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
		for k in tqdm(range(400)):
			x0 = [400*np.random.uniform(size=4) - 200]
			sol = optimize.root(F, x0, method='hybr')
			if not np.isclose(sol.x, S, atol=self.atol, rtol=self.rtol).any():
				if np.isclose(F(sol.x), [0., 0., 0., 0.], atol=self.atol, rtol=self.rtol).all():
					S.append(sol.x)
		return S


	def postproc(self, tsr, S, critic, info, reject):
		"""
		
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
		def med(t):
			if torch.numel(t) == 0:
				return 0.
			else:
				return torch.median(t).item()
		
		if reject and len(S)!=3:
			if self.info : print("Model not bistable")
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		# Sorting neural activities wrt to H/L states.
		tsr_ = tsr.narrow(1, self.smin, self.smax)
		
		# TRY AND UNDERSTAND THIS PART
		mask_he = torch.gt(tsr_[self.RE,:], critic[0].item())
		mhe = med(tsr_[self.RE][mask_he])
		md_he = (torch.sum(mask_he, dtype=self.enc).item() * self.dt) / (self.window*self.usf)
		
		if self.reject and np.isclose(md_he, 0., atol=self.atol, rtol=self.rtol):
			if self.info == True:
				print('\n Simulated dynamics never reached high activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)
		
		mask_le = torch.le(tsr_[self.RE,:], critic[0].item())
		mle = med(tsr_[self.RE][mask_le])
		md_le = (torch.sum(mask_le, dtype=self.enc).item() * self.dt) / (self.window*self.usf)
		
		if reject and np.isclose(md_le, 0., atol=self.atol, rtol=self.rtol):
			if self.info: print('\n Simulated dynamics never reached low activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		
		mask_hp, mask_hs, mask_hv = torch.gt(tsr_[self.RP,:], critic[1].item()), torch.gt(tsr_[self.RS,:], critic[2].item()), torch.gt(tsr_[self.RV,:], critic[3].item())
		mhp, mhs, mhv = med(tsr_[self.RP][mask_hp]), med(tsr_[self.RS][mask_hs]), med(tsr_[self.RV][mask_hv])

		# Reject if saturation
		if self.reject and not all(activity < .45 for activity in [mhe / self.Ae, mhp / self.Ap, mhs / self.As, mhv/self.Av]) :
			if self.info: print('\n Simulation has been terminated due to saturated dynamics.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc) 
		
		# Reject if no high activity state 
		if self.reject and np.isclose(md_he, 0., atol=self.atol, rtol=self.rtol):
			if self.info: print('\n Simulated dynamics never reached high activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)
		
		mask_lp, mask_ls, mask_lv = torch.le(tsr_[self.RP,:], critic[1].item()), torch.le(tsr_[self.RS,:], critic[2].item()), torch.le(tsr_[self.RV,:], critic[3].item())
	
		# Median activities in H/L states
		me, mle = med(tsr_[self.RE,:]), med(tsr_[self.RE][mask_le])
		mp, mlp = med(tsr_[self.RP,:]), med(tsr_[self.RP][mask_lp])
		ms, mls = med(tsr_[self.RS,:]), med(tsr_[self.RS][mask_ls])
		mv, mlv = med(tsr_[self.RV,:]), med(tsr_[self.RV][mask_lv])
   
		# Summary statistics of simulated data
		return torch.as_tensor([me, mp, ms, mv, mle, mlp, mls, mlv, mhe, mhp, mhs, mhv, md_le, md_he], device=self.dev, dtype=self.enc)

		'''



	## Maybe add an attribute which is a list of all ran simulations ? So that a model keeps a memory of all its simulations. 
	# This way we return simulation stats and data but also store them in the class

	eqs = get_equilibria()
	tsr, stim = get_simulated_data(self.plot)
	critic = torch.as_tensor(sort(eqs, 0)[1], device=self.dev, dtype=self.enc)
	res = post_processing(tsr, eqs, critic, info, reject)
		
	if info == True:
		print('\n Tested parameters: \n\n', dof,
			  '\n\n Simulation window [s]: ', window, ' ; Time resolution [s]:', dt, ' ; Refractory period [dt]: ', tr,
			  '\n\n Number of stimuli: ', len(stim), ' ; Data shape: ', tsr.size(),
			  '\n\n Number of equilibria: ', len(eqs),
			  '\n\n Equilibria: \n\n', sort(eqs, 0),
			  '\n\n Summary statistics of simulated data: \n\n', res, '\n')

	if plot == True:
		return tsr, stim, eqs, critic, res
	else:
		return res

'''