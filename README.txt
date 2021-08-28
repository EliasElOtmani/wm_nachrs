
############
# ABSTRACT #
############

This project corresponds to a research work performed at the Laboratoy of Cognitive and Computational Neuroscience (LNC2) of the Ecole Normale Supérieure of Paris, in the Team Mathematics of Neural Circuits of Pr. Boris Gutkin. 
The corresponding thesis is included in the folder. 

###########
# SCRIPTS #
###########

This code is almost entirely object-oriented. The four important classes are : 

-----------------------------------------------------------------------------------------------------
* NeuralPop.py :	Stands for a Neural Population, which also includes the nested class Synapse. This class functions as an input-output object which can be interconnected with other Neural Populations.

* CCM.py : 	Canonical Cortical Model, this class stands for a model of supragranular prefrontal circuitry. It uses interconnected Neural Populations and generates (and stores) Simulations.

* Simulation.py :	Recording of simulated neural traces over time. Its postproc function allows to return the corresponding summary statistics. 

* DMTS.py : 	The Delayed Match-To-Sample task, it uses CCMs and generates Simulations according to defined task parameters. 
-----------------------------------------------------------------------------------------------------

Most of the heavy computations (i.e., simulations) were performed on the parallel computing cluster of the LNC2, "Frontex". The cluster runs on functions defined on the following scripts :  

-----------------------------------------------------------------------------------------------------
* dmts_frontex.py

* helper_frontex.py :	These two scripts generate simulations for DMTS analysis

* infer_frontex.py : 	For simulating data with random models for further inference
-----------------------------------------------------------------------------------------------------

Finally, the repository includes the following Jupyter Notebooks : 

-----------------------------------------------------------------------------------------------------
* DMTS_analysis.ipynb 		Analysis of DMTS simulations 

* infer_on_presim_data.ipynb 	Perform Approximate Bayesian Computation

* div_modulation.ipynb 		Study of neural subpopulation response functions
-----------------------------------------------------------------------------------------------------

########
# DATA #
########

Simulated output data is stored in the brand_new_data and inference_data repositories 

