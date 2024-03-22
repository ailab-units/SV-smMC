
from MAPK import *
from BooleanLabeler import *
import os
from smt.sampling_methods import LHS
import pickle
from tqdm import tqdm


# Set the dimension

nb_params = 22
nb_reactions = 10

results_fld = "../../data/MAPK/"
os.makedirs(results_fld, exist_ok = True)

#5000
n_param_values = 100*nb_reactions # this should grow linearly with the dimension
n_trajs_per_param = 1000

nsteps = 32
timestep = 60
timeline = np.arange(nsteps+1)
final_time = nsteps*timestep
# Set of 2 STL properties
# the 2 properties have the same structure but for each model
# we randomly sample different species to be monitored

param_bounds = np.array([[0.1, 2.5], #V1
				[1, 2], #n
				[8, 10], #Kl				
				[9, 11], #K1
				[0.1, 0.4], #V2
				[7, 9], #K2
				[0.01, 0.04], #k3
				[14, 16], #K3
				[0.01, 0.04], #k4
				[14, 16], #K4
				[0.5, 1], #V5
				[14, 16], #K5
				[0.5, 1], #V6
				[14, 16], #K6
				[0.01, 0.04], #k7
				[14, 16], #K7
				[0.01, 0.04], #k8
				[14, 16], #K8
				[0.25, 0.75], #V9
				[14, 16], #K9
				[0.25, 0.75], #V10
				[14, 16], #K10		
				])

model_name = 'MAPK'
log_param_bounds = np.hstack((np.log(param_bounds[:,1:]),np.log(param_bounds[:,0:1])))
log_sampling_fnc = LHS(xlimits=log_param_bounds) #should sample logarithmically

stl_property = f'F_[0,{final_time}](G_[0,{final_time}] (H > 100) ) '

nb_test_trajs = 20


crn = MAPK(nsteps, timestep)
labeler = BooleanLabeler(model_name, results_fld, crn, n_param_values, n_trajs_per_param)

fn = f"../../data/MAPK/MAPK_DS_{n_param_values}samples_{n_trajs_per_param}obs_bis.pickle"



# parameters sampled according to a latin strategy	
sampled_parameters = np.exp(log_sampling_fnc(n_param_values))

labels = np.empty((n_param_values, n_trajs_per_param))
	
# Generate SSA trajectories for each set of parameter values
for j in tqdm(range(n_param_values)):
	crn.set_parameter_values(sampled_parameters[j])
	ssa_trajs_j = crn.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)
	
	#fig = plt.figure()

	for k in range(n_trajs_per_param):
		labels[j,k] = labeler.analyze(ssa_trajs_j[k],stl_property)
		
		#plt.plot(timeline, ssa_trajs_j[k]['H'],c='b')
		
	#plt.savefig(results_fld+f'plots/H_trajs_point_{j}')
	#plt.close()

print("Satisfaction: ", np.mean(labels, axis=1))
dataset_dict = {"params": sampled_parameters, "labels": labels, "formulas": stl_property}
labeler.save_dataset_dict(dataset_dict)

	
flag = False

while flag:
	test_labels = np.empty((nb_test_trajs, n_trajs_per_param))

	# Generate SSA trajectories for each set of parameter values
	for j in tqdm(range(nb_test_trajs)):
		crn.set_parameter_values(sampled_parameters[j])
		ssa_trajs_j = crn.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)

		for k in range(n_trajs_per_param):		
			test_labels[j,k] = labeler.analyze(ssa_trajs_j[k],stl_property)
