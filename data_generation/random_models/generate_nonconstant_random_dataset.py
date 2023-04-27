from utils import *
from RandomCRN import *
from BooleanLabeler import *
import os
from smt.sampling_methods import LHS
import pickle
from tqdm import tqdm
from rnd_gen_utils import *


# Set the dimension
n_reactions = 4
print("DIM: ", n_reactions)
max_n_species = n_reactions+1

results_fld = "../../data/Dim{}".format(n_reactions)
os.makedirs(results_fld, exist_ok = True)

pop_ub = 10

n_models = 5

n_param_values = 5000*n_reactions # this should grow linearly with the dimension
n_trajs_per_param = 10 

final_time = 10

# Set of 2 STL properties
# the 2 properties have the same structure but for each model
# we randomly sample different species to be monitored

param_bounds = [0.001,1]
log_param_bounds = [np.log(param_bounds[1]),np.log(param_bounds[0])]*np.ones((n_reactions,2))
log_sampling_fnc = LHS(xlimits=log_param_bounds) #should sample logarithmically

nb_test_trajs = 20

for i in range(n_models):
	print("---- Model {} ------------------".format(i))
	model_name_i = "NCM"+str(i)
	
	# Randomly sample a CRN
	reactions_i, selected_species_i = gen_rnd_reactions(max_n_species, n_reactions)
	print("Selected species: ", selected_species_i, "nb of selected species: ", len(selected_species_i))
	initial_state_i = np.random.randint(low=0, high = pop_ub, size=len(selected_species_i))
	print('random initial state (fixed) --> ', initial_state_i)

	crn_i = RandomCRN(model_name_i, selected_species_i, reactions_i, initial_state_i)
	#crn_i.print_crn_eqs()
	crn_i.print_crn_reactions()
	labeler_i = BooleanLabeler(model_name_i, results_fld, crn_i, n_param_values, n_trajs_per_param)
	
	nb_properties = 3

	properties_i = gen_rnd_properties(selected_species_i, final_time, pop_ub)
	# parameters sampled according to a latin strategy	
	sampled_parameters_i = np.exp(log_sampling_fnc(n_param_values))
	
	test_labels_i = np.empty((nb_test_trajs, n_trajs_per_param, nb_properties))
	
	flag = True

	while flag:
		# Generate SSA trajectories for each set of parameter values
		for j in tqdm(range(nb_test_trajs)):
			crn_i.set_parameter_values(sampled_parameters_i[j])
			ssa_trajs_j = crn_i.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)

			for k in range(n_trajs_per_param):
				for p in range(nb_properties):
					test_labels_i[j,k,p] = labeler_i.analyze(ssa_trajs_j[k],properties_i[p])

		flag = False
		for p in range(nb_properties):
			if (np.mean(test_labels_i[:,:,p]) == 1) or (np.mean(test_labels_i[:,:,p]) == 0):
				print(p)
				properties_i[p] = regen_rnd_property(p,final_time,selected_species_i, pop_ub)
				flag = True

	for pp in range(nb_properties):
		print(model_name_i+": STL formula {pp} = ", properties_i[pp])
	
	labels_i = np.empty((n_param_values, n_trajs_per_param, nb_properties))
		
	# Generate SSA trajectories for each set of parameter values
	for j in tqdm(range(n_param_values)):
		crn_i.set_parameter_values(sampled_parameters_i[j])
		ssa_trajs_j = crn_i.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)

		for k in range(n_trajs_per_param):
			for p in range(nb_properties):
				labels_i[j,k,p] = labeler_i.analyze(ssa_trajs_j[k],properties_i[p])

	print("Satisfaction: ", np.mean(labels_i, axis=1))
	dataset_dict_i = {"params": sampled_parameters_i, "labels": labels_i, "crn": crn_i, "formulas": properties_i, "initial_state": initial_state_i}
	labeler_i.save_dataset_dict(dataset_dict_i)


