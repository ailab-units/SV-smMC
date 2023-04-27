import os
import numpy as np
from settings import *
import pickle5 as pickle
from data_utils import get_tensor_data, normalize_columns


first = True
for filepath, train_filename, val_filename, params_list, math_params_list in case_studies_unc:

    print(filepath)
    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    x = val_data['params']
    if first:
        y_bern = val_data['labels']
        first = False
    else:
        y_bern = np.concatenate((y_bern,val_data['labels']),axis=0)
    
sample_variance = np.array([((param_y-param_y.mean(0))**2).mean(0) for param_y in y_bern])
std = np.sqrt(sample_variance)
    
n_trials_val = get_tensor_data(val_data)[3]
errors = (1.96*std)/np.sqrt(n_trials_val)
test_unc = 2*errors
print(test_unc.shape)
print(test_unc.mean(), 1.96*test_unc.std()/3)