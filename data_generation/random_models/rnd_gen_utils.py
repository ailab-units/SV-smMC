import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

def plot_trajectories(results, n_trajs, crn, fld, idx):
    fig = plt.figure()
    colors = ['b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w']
    for k in range(n_trajs):
	    for i in range(crn.n_species):
	        plt.plot(results[k]['time'], results[k][crn.species_names[i]], c=colors[i])
    new_fld = fld+"/"+crn.name+"_plots/"
    os.makedirs(new_fld, exist_ok = True)
    plt.title(str(idx))
    plt.savefig(new_fld+'trajs_random_crn_point={}.png'.format(idx))
    plt.close()


def globally_formula(final_time, species):
	formula = 'G_[{},{}](S{}-S{}<0)'.format(final_time/2,final_time, species[0], species[1])
	return formula

def eventually_formula(final_time, species):
	formula = 'F_[0,{}](S{}-S{}>0)'.format(final_time, species[0], species[1])
	return formula

def eventually_globally_formula(final_time, species, pop_ub):
	formula = 'F_[0,{}](G_[0,{}](S{}<{}))'.format(final_time,final_time,species[0],2*pop_ub)
	return formula

def gen_rnd_properties(selected_species, final_time, pop_ub):
	spec_one = np.random.choice(selected_species, size=2, replace=False)
	spec_two = np.random.choice(selected_species, size=2, replace=False)
	spec_three = np.random.choice(selected_species, size=1, replace=False)

	# Randomly defining two stl properties
	formula_one = globally_formula(final_time,spec_one)
	formula_two = eventually_formula(final_time,spec_two)
	formula_three = eventually_globally_formula(final_time,spec_three,pop_ub)
	
	properties = [formula_one,formula_two, formula_three]

	return properties 

def regen_rnd_property(ind, final_time, selected_species, pop_ub):
	species = np.random.choice(selected_species, size=2, replace=False)
	if ind//2 == 0:
		formula = globally_formula(final_time,species)
	elif ind//2 == 1:
		formula = eventually_formula(final_time,species)
	else:
		formula = eventually_globally_formula(final_time,species[0],pop_ub)

	return formula