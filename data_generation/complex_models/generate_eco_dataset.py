
from Eco import *
from BooleanLabeler import *
import os
from smt.sampling_methods import LHS
import pickle
from tqdm import tqdm



# Set the dimension

nb_params = 6
nb_reactions = 6

results_fld = "../../data/Eco/"
os.makedirs(results_fld, exist_ok = True)

#5000
n_param_values = 5000*nb_reactions # this should grow linearly with the dimension
n_trajs_per_param = 20

nsteps = 32
timestep = 0.1
timeline = np.arange(nsteps+1)
final_time = nsteps*timestep
# Set of 2 STL properties
# the 2 properties have the same structure but for each model
# we randomly sample different species to be monitored

param_bounds = np.array([[0.1, 2], #k1
				[0.1, 2], #k2
				[0.1, 2], #k3				
				[0.001, 0.2], #k4
				[0.001, 0.2],#k5
				[0.005, 0.5], #k6
				])

model_name = 'Eco'
log_param_bounds = np.hstack((np.log(param_bounds[:,1:]),np.log(param_bounds[:,0:1])))
log_sampling_fnc = LHS(xlimits=log_param_bounds) #should sample logarithmically

stl_property = f'F_[0,{final_time}](G_[0,{final_time}] (A > 50) ) '



crn = Eco(nsteps, timestep)
labeler = BooleanLabeler(model_name, results_fld, crn, n_param_values, n_trajs_per_param)
	
#fn = f"../../data/Eco/Eco_DS_{n_param_values}samples_{n_trajs_per_param}obs.pickle"



# parameters sampled according to a latin strategy	
sampled_parameters = np.exp(log_sampling_fnc(n_param_values))

print(sampled_parameters)
labels = np.empty((n_param_values, n_trajs_per_param))
	
# Generate SSA trajectories for each set of parameter values
for j in tqdm(range(n_param_values)):
	crn.set_parameter_values(sampled_parameters[j])
	ssa_trajs_j = crn.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)
	
	#fig,ax = plt.subplots(3)

	for k in range(n_trajs_per_param):
		labels[j,k] = labeler.analyze(ssa_trajs_j[k],stl_property)
		
		#ax[0].plot(timeline, ssa_trajs_j[k]['A'],c='b')
		#ax[1].plot(timeline, ssa_trajs_j[k]['B'],c='r')
		#ax[2].plot(timeline, ssa_trajs_j[k]['C'],c='g')
		
	#plt.tight_layout()
	#plt.savefig(results_fld+f'plots/trajs_point_{j}')
	#plt.close()

print("Satisfaction: ", np.mean(labels, axis=1))
dataset_dict = {"params": sampled_parameters, "labels": labels, "formulas": stl_property}

labeler.save_dataset_dict(dataset_dict)

	

