#!/usr/bin/python

import numpy as np
from numpy import nan
from scipy import optimize
import os, sys, time, torch
from tqdm import tqdm
import matplotlib.pyplot as plt 
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

default_task_prm = [5, 1, (1.850,1.950), (2.65, 2.75), (3.65,3.75)]
dt = 0.01

class DMTS():

	def __init__(self, ccm, task_prm = default_task_prm, sim_prm = None, reject = False):
		
		computer = 'cpu'
		self.enc = torch.float64
		self.dev = torch.device(computer)

		self.ccm = ccm
		self.sim_prm = sim_prm 

		self.nb_trials = task_prm[0]
		self.t_stim = task_prm[1]			# Stimulus timing
		self.load_interval = task_prm[2]
		self.delay_interval = task_prm[3]
		self.clear_interval = task_prm[4]

		self.trials = []

		try : self.critic = torch.as_tensor(ccm.S[-2], device=self.dev, dtype=self.enc)
		except : 
			self.critic = torch.as_tensor([10,10,10,10], device = self.dev, dtype = self.enc) # Maybe add an equivalent of info
		critic_re = self.critic[0].item()

		# Successful operations : 
		self.loadings = 0
		self.maintenances = 0
		self.clearances = 0

		for trial in range(self.nb_trials):
			
			sim = ccm.simulate(sim_prm, dmts = True, cue_timings = [self.t_stim], reject = reject)

			load_fr = sim.traces[0].narrow(0, int(self.load_interval[0]/dt), int((self.load_interval[1] - self.load_interval[0])/dt))
			delay_fr = sim.traces[0].narrow(0, int(self.delay_interval[0]/dt), int((self.delay_interval[1] - self.delay_interval[0])/dt))
			clear_fr = sim.traces[0].narrow(0, int(self.clear_interval[0]/dt), int((self.clear_interval[1] - self.clear_interval[0])/dt))

			if torch.gt(load_fr, critic_re).all() : self.loadings += 1
			if torch.gt(delay_fr, critic_re).all() : self.maintenances += 1
			if torch.gt(clear_fr, critic_re).all() : self.clearances += 1

			self.trials.append(sim)

	def print_stats(self):
		print('Number of trials :\t', self.nb_trials,
			'\nSuccessful operations :\t', self.loadings, '\t', self.maintenances, '\t', self.clearances,
			'\nRatios :\t\t', round(self.loadings/self.nb_trials, 2), '\t', round(self.maintenances/self.nb_trials, 2), '\t', round(self.clearances/self.nb_trials, 2))

	def plot_trials(self, trial_indexes = 0):	## FIX THE TRIAL INDEXES	

		dt = 0.001

		fig, ax = plt.subplots()
		ax.plot([10*i for i in range(len(self.trials[trial_indexes].traces[0]))], self.trials[trial_indexes].traces[0], color = 'r')
		#ax.vlines(x = [int(i/dt) for i in self.load_interval], ymin  = 0, ymax = 80, color = 'o') 
		#ax.vlines(x = [int(i/dt) for i in self.delay_interval], ymin  = 0, ymax = 80, color = 'b') 
		#ax.vlines(x = [int(i/dt) for i in self.clear_interval], ymin  = 0, ymax = 80, color = 'g') 
		ax.hlines(y = self.critic[0].item(), xmin = 0, xmax = 10*len(self.trials[0].traces[0]), linestyle = 'dashed')
		ax.axvspan(self.t_stim/dt, self.load_interval[1]/dt, facecolor = 'orange', alpha = 0.2)
		ax.axvspan(self.load_interval[1]/dt, self.delay_interval[1]/dt, facecolor = 'b', alpha = 0.2)
		ax.axvspan(self.delay_interval[1]/dt, self.clear_interval[1]/dt, facecolor = 'g', alpha = 0.2)
		ax.set_xlim(left = int(self.t_stim/dt/2), right = int(self.clear_interval[1]/dt + self.t_stim/dt/2))
		ax.set_ylim(top = 90)
		ax.text(self.t_stim/dt +10, 85, "Load", fontsize=13)
		ax.text(self.load_interval[1]/dt +10, 85, "Delay", fontsize=13)
		ax.text(self.delay_interval[1]/dt +10, 85, "Clear", fontsize=13)

		# Make the title bigger
		ax.set(title = 'DMTS : example trial', xlabel = 'Time (ms)', ylabel = 'PYR firing rate (Hz)')
		plt.show()



