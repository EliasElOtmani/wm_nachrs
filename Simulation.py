import numpy as np
from numpy import nan
import os, sys, time, torch


class Simulation():

	def __init__(self, ccm, traces, stimuli, sim_prm, reject=False, info=False, plot=False, dmts = False, aborted = False, critic = torch.as_tensor([20,20,20,20])):

		computer = "cpu"
		self.enc = torch.float64
		self.dev = torch.device(computer)

		self.aborted = aborted
		self.critic = critic

		self.ccm = ccm 

		# Model parameters we're interested in
		self.tau, self.tau_adp = ccm.mod_prm[0].item(), ccm.mod_prm[1].item()
		self.tr = ccm.mod_prm[13].item() # Refractory time [in unit 'dt'](float).
		self.Ae, self.Ap, self.As, self.Av = ccm.dof[0].item(), ccm.dof[1].item(), ccm.dof[2].item(), ccm.dof[3].item()

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

		#self.usf = dof[21].item()
		self.usf = ccm.mod_prm[12].item()

		# Short-hand indices for tensor-based data storage [1](int)

		if not ccm.two_columns : self.RE, self.RP, self.RS, self.RV, self.TT = 0, 1, 2, 3, 4
		else : self.RE, self.RP, self.RS, self.RV, self.RE2, self.RS2, self.RV2, self.TT = 0, 1, 2, 3, 4, 5, 6, 7
		#self.RE, self.RP, self.RS, self.RV, self.dRE, self.dRP, self.dRS, self.dRV, self.TT = 0, 1, 2, 3, 4, 5, 6, 7, 8
		#self.INE_E, self.INE_P, self.INE_S, self.INE_V, self.INS_E, self.INS_P, self.INS_S, self.INS_V = 9, 10, 11, 12, 13, 14, 15, 16
		#self.IND_E, self.IE_ADP = 17, 18


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
		

		if self.aborted :
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)  

		# Sorting neural activities wrt to H/L states #
		###############################################

		tsr_ = self.traces.narrow(1, self.smin, self.smax-self.smin)	# We get rid of neural traces outside our simulation window's scope
		
		mask_he = torch.gt(tsr_[self.RE,:], self.critic[0].item())	# Boolean vector corresponding to each PYR fr value being above (True) or below (False) critical point 
		mask_le = ~ mask_he
		
		# Statistics of H- & L- state durations (means H & L, mode H)  
		h_chunks = []
		l_chunks = []

		h_terminated = mask_he[0]
		l_terminated = ~ h_terminated
		h_length = 0
		l_length = 0

		for k in mask_he:
			if k:
				if l_terminated :
					l_chunks.append(l_length*self.dt)
					l_length = 0
					l_terminated = False
				h_length += 1
				h_terminated = True
			else : 
				if h_terminated : 
					h_chunks.append(h_length*self.dt)
					h_length = 0
					h_terminated = False 
				l_length += 1
				l_terminated = True

		if mask_he[-1] and len(h_chunks) == 0 : h_chunks.append(h_length*self.dt) 
		elif not mask_he[-1] and len(l_chunks) == 0 : l_chunks.append(l_length*self.dt) # We don't consider last chunks unless unique

		h_hist = np.histogram(h_chunks, bins = [0.25*i for i in range(40)])
		h_mode_duration = h_hist[1][:-1][h_hist[0] == max(h_hist[0])]
		h_mode_duration = h_mode_duration[0]

		h_nb = len(h_chunks) 	# Nb of H-states
		l_nb = len(l_chunks)

		if h_nb != 0 : md_he = np.mean(h_chunks)
		else : md_he = nan 
		if l_nb != 0 : md_le = np.mean(l_chunks)
		else : md_le = nan
 
		mhe = med(tsr_[self.RE][mask_he])	# Median PYR H-state FR
		mle = med(tsr_[self.RE][mask_le])
		if self.ccm.two_columns : 
			mhe2 = med(tsr_[self.RE2][mask_he])
			mle2 = med(tsr_[self.RE2][mask_le])

		if self.reject and np.isclose(md_he, 0., atol=self.atol, rtol=self.rtol):
			if self.info : print('\n Simulated dynamics never reached high activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		if self.reject and np.isclose(md_le, 0., atol=self.atol, rtol=self.rtol):
			if self.info: print('\n Simulated dynamics never reached low activity state.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		#mask_hp, mask_hs, mask_hv = torch.gt(tsr_[self.RP,:], self.critic[1].item()), torch.gt(tsr_[self.RS,:], self.critic[2].item()), torch.gt(tsr_[self.RV,:], self.critic[3].item())
		#mhp, mhs, mhv = med(tsr_[self.RP][mask_hp]), med(tsr_[self.RS][mask_hs]), med(tsr_[self.RV][mask_hv]) 
		
		#mask_lp, mask_ls, mask_lv = torch.le(tsr_[self.RP,:], self.critic[1].item()), torch.le(tsr_[self.RS,:], self.critic[2].item()), torch.le(tsr_[self.RV,:], self.critic[3].item())
	
		'''
		# Median activities in H/L states
		me, mle = med(tsr_[self.RE,:]), med(tsr_[self.RE][mask_le])
		mp, mlp = med(tsr_[self.RP,:]), med(tsr_[self.RP][mask_lp])
		ms, mls = med(tsr_[self.RS,:]), med(tsr_[self.RS][mask_ls])
		mv, mlv = med(tsr_[self.RV,:]), med(tsr_[self.RV][mask_lv])
		'''

		# Median activities in H/L states
		mhp, mlp = med(tsr_[self.RP,mask_he]), med(tsr_[self.RP][mask_le])
		mhs, mls = med(tsr_[self.RS,mask_he]), med(tsr_[self.RS][mask_le])
		mhv, mlv = med(tsr_[self.RV,mask_he]), med(tsr_[self.RV][mask_le])
		if self.ccm.two_columns : 
			mhs2, mls2 = med(tsr_[self.RS2,mask_he]), med(tsr_[self.RS2][mask_le])
			mhv2, mlv2 = med(tsr_[self.RV2,mask_he]), med(tsr_[self.RV2][mask_le])

		# Reject if saturation
		if self.reject and not all(activity < .45 for activity in [mhe / self.Ae, mhp / self.Ap, mhs / self.As, mhv/self.Av]) :
			if self.info: print('\n Simulation aborted due to saturated dynamics.')
			return torch.as_tensor([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan], device=self.dev, dtype=self.enc)

		# Summary statistics of simulated data 		
		# return torch.as_tensor([me, mp, ms, mv, mle, mlp, mls, mlv, mhe, mhp, mhs, mhv, md_le, md_he, h_mode_duration, l_nb, h_nb], device=self.dev, dtype=self.enc)
		if not self.ccm.two_columns : return torch.as_tensor([mle, mlp, mls, mlv, mhe, mhp, mhs, mhv, md_le, md_he, h_mode_duration], device=self.dev, dtype=self.enc)
		else : return torch.as_tensor([mle, mlp, mls, mlv, mle2, mls2, mlv2, mhe, mhp, mhs, mhv, mhe2, mhs2, mhv2], device=self.dev, dtype=self.enc)

	def plot(self):
		pass


