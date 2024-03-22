import numpy as np
import gillespy2
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

class MAPK(gillespy2.Model):
    def __init__(self, nsteps, timestep):
        # First call the gillespy2.Model initializer.
        super().__init__(self)

        # Define parameters for the rates of creation and dissociation.
        V1 = gillespy2.Parameter(name='V1', expression=0.1)
        n = gillespy2.Parameter(name='n', expression=1)
        Kl = gillespy2.Parameter(name='Kl', expression=9)
        K1 = gillespy2.Parameter(name='K1', expression=10)
        V2 = gillespy2.Parameter(name='V2', expression=0.25)
        K2 = gillespy2.Parameter(name='K2', expression=8)
        k3 = gillespy2.Parameter(name='k3', expression=0.025)
        K3 = gillespy2.Parameter(name='K3', expression=15)
        k4 = gillespy2.Parameter(name='k4', expression=0.025)
        K4 = gillespy2.Parameter(name='K4', expression=15)
        V5 = gillespy2.Parameter(name='V5', expression=0.75)
        K5 = gillespy2.Parameter(name='K5', expression=15)
        V6 = gillespy2.Parameter(name='V6', expression=0.75)
        K6 = gillespy2.Parameter(name='K6', expression=15)
        k7 = gillespy2.Parameter(name='k7', expression=0.025)
        K7 = gillespy2.Parameter(name='K7', expression=15)
        k8 = gillespy2.Parameter(name='k8', expression=0.025)
        K8 = gillespy2.Parameter(name='K8', expression=15)
        V9 = gillespy2.Parameter(name='V9', expression=0.5)
        K9 = gillespy2.Parameter(name='K9', expression=15)
        V10 = gillespy2.Parameter(name='V10', expression=0.5)
        K10 = gillespy2.Parameter(name='K10', expression=15)
        self.params_list = [V1,n,Kl,K1,V2,K2,k3,K3,k4,K4,V5,K5,V6,K6,k7,K7,k8,K8,V9,K9,V10,K10]
        self.add_parameter(self.params_list)

        # Define variables for the molecular species representing M & D.
        A = gillespy2.Species(name='A', initial_value=50)
        B = gillespy2.Species(name='B', initial_value=50)
        C = gillespy2.Species(name='C', initial_value=100)
        D = gillespy2.Species(name='D', initial_value=100)
        E = gillespy2.Species(name='E', initial_value=100)
        F = gillespy2.Species(name='F', initial_value=100)
        G = gillespy2.Species(name='G', initial_value=100)
        H = gillespy2.Species(name='H', initial_value=100)
        
        self.species_names = ['A','B','C','D','E','F','G','H']
        self.species_list = [A,B,C,D,E,F,G,H]
        self.add_species(self.species_list)

        # The list of reactants and products for a Reaction object are
        # each a Python dictionary in which the dictionary keys are
        # Species objects and the values are stoichiometries of the
        # species in the reaction.
        r1 = gillespy2.Reaction(name="r1", propensity_function='V1*A/( (1+(H/Kl)**n)*(K1+A) )',
                                 reactants={A:1}, products={B:1})

        r2 = gillespy2.Reaction(name="r2", propensity_function='V2*B/(K2+B)',
                                 reactants={B:1}, products={A:1})
        
        r3 = gillespy2.Reaction(name="r3", propensity_function='k3*B*C/(K3+C)',
                         reactants={C:1}, products={D:1})

        r4 = gillespy2.Reaction(name="r4", propensity_function='k4*B*D/(K4+D)',
                                 reactants={D:1}, products={E:1})
        
        r5 = gillespy2.Reaction(name="r5", propensity_function='V5*E/(K5+E)',
                                 reactants={E:1}, products={D:1})
        
        r6 = gillespy2.Reaction(name="r6", propensity_function='V6*D/(K6+D)',
                         reactants={D:1}, products={C:1})

        r7 = gillespy2.Reaction(name="r7", propensity_function='k7*E*F/(K7+F)',
                                 reactants={F:1}, products={G:1})
        
        r8 = gillespy2.Reaction(name="r8", propensity_function='k8*E*G/(K8+G)',
                                 reactants={G:1}, products={H:1})
        
        r9 = gillespy2.Reaction(name="r9", propensity_function='V9*H/(K9+H)',
                         reactants={H:1}, products={G:1})

        r10 = gillespy2.Reaction(name="r10", propensity_function='V10*G/(K10+G)',
                                 reactants={G:1}, products={F:1})
        
        self.add_reaction([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10])

        # Set the timespan for the simulation.
        self.timespan(np.arange(start=0, stop=(nsteps+1)*timestep, step=timestep))


    #def set_obs_state(self, new_state):
        
    #    for i in range(self.n_species):
    #        self.species_list[i].initial_value = new_state[i]

    def set_parameter_values(self, new_param_values):
        
        for i in range(len(self.params_list)):
            self.params_list[i].expression = new_param_values[i]

