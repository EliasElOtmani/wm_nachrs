import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch


class Simulation():

	def __init__(self, traces, stimuli, sim_prm, mod_prm, dof, S, reject=False, info=False, plot=False, dmts = False, aborted = False):

		computer = "cpu"
		self.enc = torch.float64
		self.dev = torch.device(computer)

		self.aborted = aborted

		# Model parameters we're interested in
		self.tau, self.tau_adp = mod_prm[0].item(), mod_prm[1].item()
		self.tr = mod_prm[13].item() # Refractory time [in unit 'dt'](float).
		self.Ae, self.Ap, self.As, self.Av = dof[0].item(), dof[1].item(), dof[2].item(), dof[3].item()

		self.traces = traces
		self.stimuli = stimuli
		self.sim_prm = sim_prm

		### SIMULATION PARAMETERS
		
		self.window = sim_prm[0].item() # Stimulation window [s](float).
		self.dt = sim_prm[1].item() # Time resolution [s](float).
		self.atol, self.rtol = sim_prm[2].item(), sim_prm[3].item() # Absolute and relative tolerances for float comparison.
		self.plot = plot
		self.info = info
		self.reject = reject
		
		if torch.isnan(sim_prm[4]):
			torch.manual_seed(time.time())
		else:
			torch.manual_seed(sim_prm[4])

		self.smin = round(3 * max(self.tau, self.tau_adp) / self.dt) # Starting time for stimulation window [1](int).
		self.smax = round(self.window / self.dt) # End of stimulation window [1](int).
		self.N = self.smin + self.smax # Total number of time-points [1](int)

		self.S = S #Solutions of the system 

		#self.usf = dof[21].item()
		self.usf = mod_prm[12].item()
		# Short-hand indices for tensor-based data storage [1](int)
		self.RE, self.RP, self.RS, self.RV, self.dRE, self.dRP, self.dRS, self.dRV, self.TT = 0, 1, 2, 3, 4, 5, 6, 7, 8
		self.INE_E, self.INE_P, self.INE_S, self.INE_V, self.INS_E, self.INS_P, self.INS_S, self.INS_V = 9, 10, 11, 12, 13, 14, 15, 16
		self.IND_E, self.IE_ADP = 17, 18



	def postproc(self):
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
		

		if self.aborted or (self.reject and len(self.S)<3):
			if self.info : print("Model not bistable")
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		# 4dim vector corresponding to fr values of neural populations for critical point
		critic = torch.as_tensor(np.sort(self.S, 0)[1], device=self.dev, dtype=self.enc)

		# Sorting neural activities wrt to H/L states.
		###############################################

		tsr_ = self.traces.narrow(1, self.smin, self.smax-self.smin)	# We get rid of neural traces outside our simulation window's scope
		# CAREFUL the vector actually goes from smin to smin+smax
		
		mask_he = torch.gt(tsr_[self.RE,:], critic[0].item())	# Ouputs a boolean vector corresponding to each PYR fr value being above (True) or below (False) critical point 
		mhe = med(tsr_[self.RE][mask_he])
		md_he = (torch.sum(mask_he, dtype=self.enc).item() * self.dt) / (self.window*self.usf)
		
		if self.reject and np.isclose(md_he, 0., atol=self.atol, rtol=self.rtol):
			if self.info : print('\n Simulated dynamics never reached high activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)
		
		mask_le = torch.le(tsr_[self.RE,:], critic[0].item())
		mle = med(tsr_[self.RE][mask_le])
		md_le = (torch.sum(mask_le, dtype=self.enc).item() * self.dt) / (self.window*self.usf)
		
		if self.reject and np.isclose(md_le, 0., atol=self.atol, rtol=self.rtol):
			if self.info: print('\n Simulated dynamics never reached low activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		mask_hp, mask_hs, mask_hv = torch.gt(tsr_[self.RP,:], critic[1].item()), torch.gt(tsr_[self.RS,:], critic[2].item()), torch.gt(tsr_[self.RV,:], critic[3].item())
		mhp, mhs, mhv = med(tsr_[self.RP][mask_hp]), med(tsr_[self.RS][mask_hs]), med(tsr_[self.RV][mask_hv])

		# Reject if saturation
		if self.reject and not all(activity < .45 for activity in [mhe / self.Ae, mhp / self.Ap, mhs / self.As, mhv/self.Av]) :
			if self.info: print('\n Simulation aborted due to saturated dynamics.')
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
   
		# Summary statistics of simulated data 		# MISSING MODE H state duration 1.5s
		return torch.as_tensor([me, mp, ms, mv, mle, mlp, mls, mlv, mhe, mhp, mhs, mhv, md_le, md_he], device=self.dev, dtype=self.enc)

	def plot(self):
		pass