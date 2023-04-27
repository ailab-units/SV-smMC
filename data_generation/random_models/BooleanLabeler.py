import os
import sys
import pickle
import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) # to import pcheck

from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries


class BooleanLabeler(object):

    def __init__(self, model_name, fld, crn, n_param_values, n_trajs):

        self.crn = crn
        self.model_name = model_name
        self.fld = fld
        self.n_param_values = n_param_values
        self.n_trajs = n_trajs


    def analyze(self, experiment, formula):

        trajectories = np.stack([experiment[species] for species in self.crn.species_names])

        #forse vanno anche trasposte ste traiettorie    
        time_series = TimeSeries(self.crn.species_names, experiment['time'], trajectories) #temo che questo non funzioni
                
        label = stlBooleanSemantics(time_series, 0, formula)
        if label:
            return 1
        else:
            return 0


    def save_dataset_dict(self, dictionary):
        filename = self.fld+"{}_DS_{}samples_{}obs.pickle".format(self.model_name, self.n_param_values, self.n_trajs)

        with open(filename, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
