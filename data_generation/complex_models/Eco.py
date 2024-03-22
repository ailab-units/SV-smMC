import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from smt.sampling_methods import LHS

class Eco(gillespy2.Model):
    def __init__(self, nsteps, timestep):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        k1 = gillespy2.Parameter(name='k1', expression=0.1)
        k2 = gillespy2.Parameter(name='k2', expression=0.1)
        k3 = gillespy2.Parameter(name='k3', expression=0.1)
        k4 = gillespy2.Parameter(name='k4', expression=0.02)
        k5 = gillespy2.Parameter(name='k5', expression=0.02)
        k6 = gillespy2.Parameter(name='k6', expression=0.01)
        
        self.params_list = [k1,k2,k3,k4,k5,k6]
        self.add_parameter(self.params_list)

        # Define variables for the molecular species representing M & D.
        A = gillespy2.Species(name='A', initial_value=50)
        B = gillespy2.Species(name='B', initial_value=100)
        C = gillespy2.Species(name='C', initial_value=50)
        
        self.species_names = ['A','B','C']
        self.species_list = [A,B,C]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", propensity_function='k1*A',
                                 reactants={A:1}, products={A:2})

        r2 = gillespy2.Reaction(name="r2", propensity_function='k2*A*C/(A+C)',
                                 reactants={A:1,C:1}, products={C:1})
        
        r3 = gillespy2.Reaction(name="r3", propensity_function='k3*B*C/(B+C)',
                         reactants={B:1,C:1}, products={B:1, C:2})

        r4 = gillespy2.Reaction(name="r4", propensity_function='k4*A',
                                 reactants={A:1}, products={})
        
        r5 = gillespy2.Reaction(name="r5", propensity_function='k5*C',
                                 reactants={C:1}, products={})
        
        r6 = gillespy2.Reaction(name="r6", propensity_function='k6*A*B/(A+B)',
                         reactants={A:1,B:1}, products={})

        
        self.add_reaction([r1, r2, r3, r4, r5, r6])

        # Set the timespan for the simulation.
        self.timespan(np.arange(start=0, stop=(nsteps+1)*timestep, step=timestep))



    def set_parameter_values(self, new_param_values):
        
        for i in range(len(self.params_list)):
            self.params_list[i].expression = new_param_values[i]


if __name__ == "__main__":
    nsteps = 10
    timestep = 0.01
    timeline = np.arange(nsteps+1)
    final_time = nsteps*timestep

    n_param_values = 6 # this should grow linearly with the dimension
    n_trajs_per_param = 2 

    param_bounds = np.array([[0.1, 1], #k1
                    [0.1, 1], #k2
                    [0.5, 2], #k3               
                    [0.001, 0.2], #k4
                    [0.001, 0.2],#k5
                    [0.05, 0.5], #k6
                    ])

    model_name = 'Eco'
    log_param_bounds = np.hstack((np.log(param_bounds[:,1:]),np.log(param_bounds[:,0:1])))
    log_sampling_fnc = LHS(xlimits=log_param_bounds) #should sample logarithmically

    results_fld = "../../data/Eco/"

    crn = Eco(nsteps, timestep)
        

    # parameters sampled according to a latin strategy  
    sampled_parameters = np.exp(log_sampling_fnc(n_param_values))

    labels = np.empty((n_param_values, n_trajs_per_param))
        
    # Generate SSA trajectories for each set of parameter values
    for j in tqdm(range(n_param_values)):
        print('p = ', sampled_parameters[j])
        crn.set_parameter_values(sampled_parameters[j])

        ssa_trajs_j = crn.run(algorithm = "SSA", t= final_time, number_of_trajectories=n_trajs_per_param)
        
        fig,ax = plt.subplots(3)

        for k in range(n_trajs_per_param):
            
            ax[0].plot(timeline, ssa_trajs_j[k]['A'],c='b')
            ax[1].plot(timeline, ssa_trajs_j[k]['B'],c='r')
            ax[2].plot(timeline, ssa_trajs_j[k]['C'],c='g')
            
        plt.savefig(results_fld+f'plots/trajs_point_{j}')
        plt.close()

