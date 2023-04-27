import sys
import os
import numpy as np
import gillespy2
import matplotlib.pyplot as plt
from gillespy2 import Model, Species, Parameter, Reaction

class RandomCRN(Model):
    def __init__(self, model_name, species, reactions, init_state, timeline = np.linspace(0,10,11)):
        # Initialize the model.
        Model.__init__(self, name = model_name)
        self.n_species = len(species)
        self.reactions = reactions
        self.n_reactions = len(reactions)
        self.param_names = ['P{}'.format(i) for i in range(self.n_reactions)]
        self.reaction_names = ['R{}'.format(i) for i in range(self.n_reactions)]
        self.species_names = ['S{}'.format(i) for i in species]
        # Define parameters.
        self.params_list = []
        for j in range(self.n_reactions):
            self.params_list.append(Parameter(name = self.param_names[j], expression = 0.005))
        self.add_parameter(self.params_list)

        # Define molecular species.
        self.species_list = []
        for i in range(self.n_species):
            self.species_list.append(Species(name = self.species_names[i], initial_value = float(init_state[i])))#mode='continuous'
        self.species_list.append(Species(name = 'N0', initial_value = float(np.sum(init_state))))
        
        self.add_species(self.species_list)

        # Define reactions.
        reactions_list = []
        for k in range(self.n_reactions):
            rk = self.reactions[k]
            in_k = {}
            out_k = {}
            prop_k = self.param_names[k]
            if len(rk[0]) == 0:
                prop_k += '*N0'
            else:
                for kin in range(len(rk[0])):
                    #in_k[self.species_list[rk[0][kin]]] = 1
                    in_k['S{}'.format(rk[0][kin])] = 1
                    
                    prop_k += '*S{}'.format(rk[0][kin])
                if len(rk[0]) == 2:
                    prop_k += '/N0'

            if len(rk[1]) > 0:
                for kout in range(len(rk[1])):
                    #out_k[self.species_list[rk[1][kout]]] = 1
                    out_k['S{}'.format(rk[1][kout])] = 1

            reactions_list.append(Reaction(name = self.reaction_names[k], reactants = in_k, products = out_k,
                      propensity_function = prop_k))

        self.reactions_list = reactions_list
        self.add_reaction(reactions_list)
        self.timespan(timeline)

    def print_crn_reactions(self):
        for i in range(self.n_reactions):
            print(self.reactions_list[i])


    def set_initial_state(self, new_state):
        for i in range(self.n_species):
            self.species_list[i].initial_value = new_state[i]

    def set_parameter_values(self, new_param_values):
        for i in range(self.n_reactions):
            self.params_list[i].expression = new_param_values[i]