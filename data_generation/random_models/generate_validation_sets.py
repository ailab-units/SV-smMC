from utils import *
from RandomCRN import *
from BooleanLabeler import *
import os
from smt.sampling_methods import LHS
import pickle
from tqdm import tqdm

# Set the dimension
dim = 8
model_name = 'NCM5'
print("Dim: ", dim, "Model: ", model_name)


results_fld = "../../data/Dim{}/".format(dim)

#training set filename
fn = results_fld+model_name+'_DS_{}samples_10obs.pickle'.format(5000*dim)

file = open(fn, 'rb')
data = pickle.load(file)
file.close()

crn = data["crn"]
properties = data["formulas"]



pop_ub = 10

# VALIDATION SETTINGS
n_param_values = 100*dim # this should grow linearly with the dimension
n_trajs_per_param = 1000 

final_time = 10

param_bounds = [0.001,1]
log_param_bounds = [np.log(param_bounds[1]),np.log(param_bounds[0])]*np.ones((crn.n_reactions,2))
log_sampling_fnc = LHS(xlimits=log_param_bounds) #should sample logarithmically


labeler = BooleanLabeler(model_name, results_fld, crn, n_param_values, n_trajs_per_param)

nb_properties = len(properties)

# parameters sampled according to a latin strategy	
sampled_parameters = np.exp(log_sampling_fnc(n_param_values))

labels = np.empty((n_param_values, n_trajs_per_param, nb_properties))

# Generate SSA trajectories for each set of parameter values
for j in tqdm(range(n_param_values)):
	#print("j={}/{}".format(j+1,n_param_values))
	# paramaters are sampled according to a latin strategy
	crn.set_parameter_values(sampled_parameters[j])
	ssa_trajs_j = crn.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)

	for k in range(n_trajs_per_param):
		for p in range(nb_properties):
			labels[j,k,p] = labeler.analyze(ssa_trajs_j[k],properties[p])

dataset_dict = {"params": sampled_parameters, "labels": labels, "eqs": crn.reactions_list, "formulas": properties}
labeler.save_dataset_dict(dataset_dict)

