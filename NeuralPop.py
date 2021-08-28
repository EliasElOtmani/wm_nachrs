import numpy as np
import torch, os, math

class NeuralPop():

	def __init__(self, neural_name, ccm, amplitude, sigma, tau_m, tau_adp, a_, b_, Iext, J_adp = 0, STP = None, Tref = 1, NT = 'gaba', Itrans = 0, nAChRs = None):  #We can maybe think about pop size
		
		if NT not in ('glutamate', 'gaba', 'ACh'):
			raise ValueError(NT + ' is not supported. Please enter only glutamate or gaba as NT')

		self.ccm = ccm
		self.name = neural_name
		self.NT = NT 
		self.A = amplitude 
		self.tm = tau_m			# Mb time constant
		self.tau_adp = tau_adp	# Adaptation time constant
		self.sigma = sigma
		self.Iext = Iext
		self.J_adp = J_adp
		self.Tref = Tref

		self.a_ = a_ 
		self.b_ = b_

		self.glut_synapses = []
		self.gaba_synapses = []
		self.div_synapses = []

		self.Iadp = 0
		self.fr = 0

		self.sigma_buffer = sigma 
		self.J_adp_buffer = J_adp

		self.Itrans = Itrans*100 	# Only for thalamus      /!\ CORRECT THIS, for now we're just adding a weight of 0.1


	# Populational response function, as derived from [Papasavvas, 2015].
	def __f(self, x, b, a):	# Excitatory, Subtractive, Divisive. 
		#return .5 * ( 1. + torch.tanh( torch.as_tensor([.5 * ( (self.a_ / (1. + a)) * (x - b - self.b_) )]) ).item() ) - ( 1./(1. + torch.exp( torch.as_tensor([self.a_*self.b_/(1.+a)]) ).item() ) )
		return (self.a_/(self.a_+ a))*(1/(1 + np.exp( - (self.a_ * (x - b - self.b_)))) - 1 / (1 + np.exp(self.a_*self.b_)))

	# Asymptotic plateau of populational response functions, as derived from [Papasavvas, 2015]
	def __K(self, a):
		#return torch.exp(torch.as_tensor([self.a_*self.b_/(1.+a)])).item() / (1. + torch.exp(torch.as_tensor([self.a_*self.b_/(1.+a)])).item())
		return (np.exp(self.a_*self.b_)/(1 + np.exp(self.a_*self.b_)))*(self.a_/(self.a_ + a))

	def synapse(self, presynaptic, weight, q = 0, STP = None, stained = False, nic_normalization = False):
		if presynaptic.NT == 'gaba':
			self.gaba_synapses.append(self.Synapse(presynaptic, self, weight, q, STP, stained, nic_normalization = nic_normalization))
			if q != 0 : self.div_synapses.append(self.gaba_synapses[-1]) 	# Storing divisive syn in a specific list saves some computation time in get_derivative()
		elif presynaptic.NT in ('glutamate', 'ACh'):
			self.glut_synapses.append(self.Synapse(presynaptic, self, weight, q, STP, stained, nic_normalization = nic_normalization))
		#elif presynaptic.NT in ('ACh'):
			#self.ach_synapses.append(self.Synapse(presynaptic, weight, q, STP))


	def get_derivative(self, dt, noise = 0):	# Noise has to be sampled in or before the declaration
			
		epsp = np.sum([syn.input(dt) for syn in self.glut_synapses]) - self.Iadp + self.Iext
		sub_ipsp = np.sum([syn.input(dt) for syn in self.gaba_synapses]) 
		div_ipsp = np.sum([syn.divisive_input(dt) for syn in self.div_synapses])  
		self.dIadp = (dt/self.tau_adp) * ( - self.Iadp + self.fr * self.J_adp )
		Ke = self.__K(div_ipsp)

		self.dfr = (dt/self.tm) * ( - self.fr + (self.A*Ke - self.Tref*self.fr)*self.__f(epsp, sub_ipsp, div_ipsp) ) + Ke * np.sqrt(dt) * self.sigma * noise
		if math.isnan(self.dfr): 
			self.dfr = 0
		return self.dfr

	def step(self, dt = None, noise = None, dfr_computed = False):

		if not dfr_computed : self.get_derivative(dt, noise)
		self.Iadp += self.dIadp
		self.fr += self.dfr

	def deterministic(self): 			# Abolishes noise until reset (for equilibria computing)
		self.sigma_buffer = self.sigma 
		self.J_adp_buffer = self.J_adp 
		self.Iadp = 0
		self.sigma = 0
		self.J_adp = 0
		# /!\ GOTTA DO IT ALSO FOR THE SYNAPSES

	def reset(self):
		self.adp = 0
		self.fr = 0
		self.Iadp = 0
		self.sigma = self.sigma_buffer
		self.J_adp = self.J_adp_buffer
		for syn in self.gaba_synapses + self.glut_synapses : syn.D = 1
		

	class Synapse():  # Synapse(self) ? (--> self.Synapse())

		def __init__(self, presyn, postsyn, weight, q = 0, STP = None, stained = False, nic_normalization = False):
			self.presyn = presyn 	# Presynaptic neural population
			self.postsyn = postsyn
			self.weight = weight
			self.q = q
			self.D = 1 	# Synaptic efficacy
			self.tauD = 10 #Arbitrary 
			self.STP = STP
			self.stained = stained
			self.nic_normalization = nic_normalization

			if STP is None : self.ss2 = 1 # Steady-state of STP for infinite input current
			elif STP in ('d', 'D', 'depression', 'Depression', 'STD'): 
				if presyn.NT in ('glutamate', 'gaba') : self.ss2 = 0.5 
				elif presyn.NT == 'ACh' : self.ss2 = 0
			elif STP in ('f', 'F', 'facilitation', 'Facilitation', 'STF') : self.ss2 = 1.5
			else : 
				print('Please declare a valid synaptic plasticity such as \'D\' of \'F\'')
				os.exit()

			#self.k = 0.5*(1-self.ss2) + 1
			if STP is not None : 
				if presyn.name == 'pyr' and postsyn.name == 'pv' : self.k = 1.2328*0.5 +1   #1.5825
				elif presyn.name == 'pyr' and postsyn.name == 'som' :  self.k = 1.2328*(-0.5) +1 #0.5825
				elif presyn.name == 'pv' and postsyn.name == 'pyr' : self.k = 1.5911*0.5 +1#1.5383
				elif presyn.name == 'thalamus' and postsyn.name == 'pv' : self.k = 1
				elif presyn.name == 'cholinergic_nuclei' : self.k = 1
				else : 
					print('Using default k = 1 for unknown synaptic connexion...')
					#print(presyn.name, postsyn.name)
					self.k = 1

			# In DOWN states, wpe = 0.1, wep = 0.18, wes = .053
			# For PV outputs, k = rp*(1 - Dss) +1 = 1.0766*0.5 +1 = 1,5383
			# For inputs from PYR to PV : k = re*0.5 +1 =  1.1650*0.5 + 1 = 1.5825.
			# For inputs from Thalamus to PV : k = 1
			# For inputs from PYR to SOM : k = re*(-0.5) + 1 = -0.5825 +1 = 0.5825

		def input(self, dt):
			if self.nic_normalization : nic_mod = 1 + 0.1*(30 - self.presyn.fr) 
			else : nic_mod = 1 
			current = (1-self.q)*(self.weight*self.presyn.fr)*self.D*nic_mod
			if self.STP is not None : 	# else saves some computation
				if not math.isnan(self.presyn.fr) : self.D += ((self.k-self.D) - self.presyn.fr*(self.D - self.ss2))*(dt/self.tauD)
				else : self.D += (1-self.D)*(dt/self.tauD) 
			if self.stained and self.presyn.ccm.notebook != None : self.presyn.ccm.notebook.append(self.D*100)
			return max(current, 0.)

		def divisive_input(self, dt):
			current = self.q*self.weight*self.presyn.fr*self.D
			return max(current, 0.)

