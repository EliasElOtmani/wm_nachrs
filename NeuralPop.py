import numpy as np
import torch

class NeuralPop():

	def __init__(self, neural_name, amplitude, sigma, tau_m, tau_adp, a_, b_, Iext, J_adp = 0, STP = None, Tref = 0.01, NT = 'gaba'):  #We can maybe think about pop size
		
		if NT != 'glutamate' and NT != 'gaba':
			raise ValueError(NT + ' is not supported. Please enter only glutamate or gaba as NT')

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

		self.Iadp = 0
		self.fr = 0

		self.sigma_buffer = sigma 
		self.J_adp_buffer = J_adp


	# Populational response function, as derived from [Papasavvas, 2015].
	def __f(self, x, b, a):	# Excitatory, Subtractive, Divisive. 
		return .5 * ( 1. + torch.tanh( torch.as_tensor([.5 * ( (self.a_ / (1. + a)) * (x - b - self.b_) )]) ).item() ) - ( 1./(1. + torch.exp( torch.as_tensor([self.a_*self.b_/(1.+a)]) ).item() ) )

	# Asymptotic plateau of populational response functions, as derived from [Papasavvas, 2015]
	def __K(self, a):
		return torch.exp(torch.as_tensor([self.a_*self.b_/(1.+a)])).item() / (1. + torch.exp(torch.as_tensor([self.a_*self.b_/(1.+a)])).item())

	def synapse(self, presynaptic, weight, q = 0, STP = None):
		if presynaptic.NT == 'gaba':
			self.gaba_synapses.append(self.Synapse(presynaptic, weight, q, STP))
		elif presynaptic.NT == 'glutamate':
			self.glut_synapses.append(self.Synapse(presynaptic, weight, q, STP))


	def get_derivative(self, dt, noise = 0, trans = 0):	# Noise had to be sampled in or before the declaration
		
		Ie = self.Iext + trans # External input [1](float).		
		epsp = np.sum([syn.input() for syn in self.glut_synapses]) - self.Iadp + Ie 
		sub_ipsp = np.sum([syn.input() for syn in self.gaba_synapses]) 
		div_ipsp = np.sum([syn.divisive_input() for syn in self.gaba_synapses])
		self.dIadp = (dt/self.tau_adp) * ( - self.Iadp + self.fr * self.J_adp )
		Ke = self.__K(div_ipsp)

		self.dfr = (dt/self.tm) * ( - self.fr + (self.A*Ke - self.Tref*self.fr)*self.__f(epsp, sub_ipsp, div_ipsp) ) + self.A * Ke * np.sqrt(dt) * self.sigma * noise
		return self.dfr

	def step(self, dt, noise, trans = 0, dfr_computed = False):

		if not dfr_computed : self.get_derivative(dt, noise, trans)
		self.Iadp += self.dIadp
		self.fr += self.dfr
		self.fr = max(self.fr, 0.)

	def deterministic(self): 			# Abolishes noise until reset (for equilibria computing)
		self.sigma_buffer = self.sigma 
		self.J_adp_buffer = self.J_adp 
		self.Iadp = 0
		self.sigma = 0
		self.J_adp = 0

	def reset(self):
		self.adp = 0
		self.fr = 0
		self.Iadp = 0
		self.sigma = self.sigma_buffer
		self.J_adp = self.J_adp_buffer
		

	class Synapse():  # Synapse(self) ? (--> self.Synapse())

		def __init__(self, presyn, weight, q = 0, STP = None):
			self.presyn = presyn 	# Presynaptic neural population
			self.weight = weight
			self.q = q
			self.plasticity = STP   # We'll have to allow for fixing its weight / rate... 

		def input(self):
			return (1-self.q)*(self.weight*self.presyn.fr)

		def divisive_input(self):
			return self.q*self.weight*self.presyn.fr

