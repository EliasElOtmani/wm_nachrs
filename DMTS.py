#!/usr/bin/python

import numpy as np
from numpy import nan
#from scipy import optimize
import os, sys, time, torch
#from tqdm import tqdm
#import matplotlib.pyplot as plt 
from CCM import CCM
from Simulation import Simulation

# We want our class to have a ccm and task parameters among which 
# how many simulations to perform to obtain P(erase) etc...

'''
We want :
	- A perform function (maybe just the init ?)
	- A plot stats function : 
		- Different probabilities 
	- A plot function : 
		-Vertical line at the stimulus 

'''

# task_param = [num_trials, stimulus_timing, load_interval, delay_interval, clearance_interval]
# num_ccm with different sets of parameters ?.. Rather not

default_task_prm = [5, 1, 0.1, (1.85,1.950), (2.65, 2.75), (3.65,3.75)]
default_task_prm = [5, 1, 0.05, (0.,0.1), (2, 2.1), (3.,3.1)]
dt = 0.001

# Let's put the distractor at t = 2sec

class DMTS():

	def __init__(self, ccm, task_prm = default_task_prm, sim_prm = None, reject = False, distractor = False, distractor_amplitude = 1, distractor_timing = 2, distractor_duration = 0.2, nic_trans = False, nic_trans_amplitude = 1, nic_trans_timing = 2):
		
		computer = 'cpu'
		self.enc = torch.float64
		self.dev = torch.device(computer)

		self.ccm = ccm
		self.sim_prm = sim_prm 

		self.nb_trials = task_prm[0]
		self.t_stim = task_prm[1]			# Stimulus timing
		self.stim_duration = task_prm[2]
		self.stim_end = self.t_stim + self.stim_duration
		self.load_interval = [t + self.stim_end for t in task_prm[3]]
		self.delay_interval = [t + self.stim_end for t in task_prm[4]]
		self.clear_interval = [t + self.stim_end for t in task_prm[5]]

		self.trials = []

		try : self.critic = torch.as_tensor(ccm.S[-2], device=self.dev, dtype=self.enc)
		except : 
			self.critic = torch.as_tensor([20 for i in range(ccm.dim)], device = self.dev, dtype = self.enc) # Maybe add an equivalent of info
		critic_re = self.critic[0].item()

		# Successful operations : 
		self.loadings = 0
		self.maintenances = 0
		self.clearances = 0

		for trial in range(self.nb_trials):
			
			sim = ccm.simulate(sim_prm, dmts = True, cue_timings = [self.t_stim], Itrans_duration = self.stim_duration, reject = reject, distractor = distractor, distractor_amplitude = distractor_amplitude, distractor_timing = distractor_timing, distractor_duration = distractor_duration, nic_trans = nic_trans, nic_trans_amplitude = nic_trans_amplitude, nic_trans_timing = nic_trans_timing)

			initial_fr = sim.traces[0].narrow(0, int((self.t_stim - 0.1)/dt), int(((self.t_stim - (self.t_stim - 0.1))/dt)))					# fr over the 100 ms preceding the cue
			load_fr = sim.traces[0].narrow(0, int(self.load_interval[0]/dt), int((self.load_interval[1] - self.load_interval[0])/dt))
			delay_fr = sim.traces[0].narrow(0, int(self.delay_interval[0]/dt), int((self.delay_interval[1] - self.delay_interval[0])/dt))
			clear_fr = sim.traces[0].narrow(0, int(self.clear_interval[0]/dt), int((self.clear_interval[1] - self.clear_interval[0])/dt))

			successful_loading = False

			if torch.gt(load_fr, critic_re).all() and not torch.gt(initial_fr, critic_re).any() : 
				self.loadings += 1
				successful_loading = True
			if successful_loading and torch.gt(delay_fr, critic_re).all() : self.maintenances += 1
			if successful_loading and not torch.gt(clear_fr, critic_re).any() : self.clearances += 1

			self.trials.append(sim)

	def print_stats(self):
		print('Number of trials :\t', self.nb_trials,
			'\nSuccessful operations :\t', self.loadings, '\t', self.maintenances, '\t', self.clearances,
			'\nRatios :\t\t', round(self.loadings/self.nb_trials, 2), '\t', round(self.maintenances/self.nb_trials, 2), '\t', round(self.clearances/self.nb_trials, 2))

	def plot_trials(self, trial_indexes = 0, clear = False, save = False, save_as = None):	## FIX THE TRIAL INDEXES	

		xunit = 0.001 # Timesteps expressed in ms
		dt = 1

		fig, ax = plt.subplots()
		ax.plot([dt*i for i in range(len(self.trials[trial_indexes].traces[0]))], self.trials[trial_indexes].traces[0], color = 'r')
		#ax.vlines(x = [int(i/dt) for i in self.load_interval], ymin  = 0, ymax = 80, color = 'o') 
		#ax.vlines(x = [int(i/dt) for i in self.delay_interval], ymin  = 0, ymax = 80, color = 'b') 
		#ax.vlines(x = [int(i/dt) for i in self.clear_interval], ymin  = 0, ymax = 80, color = 'g') 
		ax.hlines(y = self.critic[0].item(), xmin = 0, xmax = dt*len(self.trials[0].traces[0]), linestyle = 'dashed')
		ax.axvspan((self.t_stim - .05)/xunit, self.load_interval[1]/xunit, facecolor = 'orange', alpha = 0.2)
		ax.axvspan(self.load_interval[1]/xunit, self.delay_interval[1]/xunit, facecolor = 'b', alpha = 0.2)
		if clear : ax.axvspan(self.delay_interval[1]/xunit, self.clear_interval[1]/xunit, facecolor = 'g', alpha = 0.2)
		ax.set_xlim(left = int(self.t_stim/xunit/2), right = int(self.clear_interval[1]/xunit + self.t_stim/xunit/2))
		ax.set_ylim(top = 90)
		ax.text(self.t_stim/xunit +10 -50, 80, "Load", fontsize=12)
		ax.text((self.load_interval[1] + (self.delay_interval[0] - self.load_interval[0])/2 - 0.25)/xunit +10, 80, "Delay", fontsize=12)
		if clear : ax.text(self.delay_interval[1]/xunit +10, 80, "Clear", fontsize=12)

		# Make the title bigger
		ax.set(title = 'DMTS : example trial', xlabel = 'Time (ms)', ylabel = 'PYR firing rate (Hz)')
		if save : plt.savefig(save_as, dpi = 300)
		plt.show()





