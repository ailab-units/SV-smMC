import os, sys
import argparse
import cProfile
import numpy as np
import pickle5 as pickle
sys.path.append(".")
from settings import *
from EP_GPs.smMC_GPEP import *
from posterior_plot_utils import plot_posterior
import RandomCRN
from data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--n_formulas", default=3, type=int, help="Nb of STL formulas")
args = parser.parse_args()

plots_path = os.path.join(plots_path, "EP_GPs/")
models_path = os.path.join(models_path, "EP_GPs/")

N_TRAINING_RUNS = 5

GEN_ERR_AVG,GEN_ERR_STD = [], []
EMP_ERR_AVG, EMP_ERR_STD = [], []
BOUND_AVG, BOUND_STD = [], []
TOT_BOUND_AVG, TOT_BOUND_STD = [], []


for filepath, train_filename, val_filename, params_list, math_params_list in stat_guar_case_studies:

    print(args)

    print(f"\n=== EP GP Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== EP GP Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)
       
    gen_err, emp_err, bound, tot_bound = [], [], [], []
    for j in range(N_TRAINING_RUNS):
        i = -1
        print(f"---formula {i}")
        out_filename = f"ep_gp_{train_filename}_formula_{i}"

        smc = smMC_GPEP()

        x_train, y_train, n_samples_train, n_trials_train = smc.transform_data_w_pindex(train_data, property_index=i)
        

        if args.load:
            smc.load(filepath=models_path, filename=out_filename, idx=str(j))

        else:
            smc.fit(x_train, y_train, n_trials_train)
            smc.save(filepath=models_path, filename=out_filename, idx=str(j))

            
        x_val, y_val, n_samples_val, n_trials_val = smc.transform_data_w_pindex(val_data, property_index=i)
        
        post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_val, y_val=val_data['labels'], 
            n_samples=n_samples_val, n_trials=n_trials_val)
        

        res_j = smc.compute_pac_bounds(x_train, y_train, x_val, y_val)

        gen_err.append(res_j[0])
        emp_err.append(res_j[1])
        bound.append(res_j[2])
        tot_bound.append(res_j[1]+res_j[2])
    
    GEN_ERR_AVG.append(np.mean(gen_err))
    GEN_ERR_STD.append(np.std(gen_err))
    EMP_ERR_AVG.append(np.mean(emp_err))
    EMP_ERR_STD.append(np.std(emp_err))
    BOUND_AVG.append(np.mean(bound))
    BOUND_STD.append(np.std(bound))
    TOT_BOUND_AVG.append(np.mean(tot_bound))
    TOT_BOUND_STD.append(np.std(tot_bound))


results_dict = {'gen': [GEN_ERR_AVG,GEN_ERR_STD],'emp':[EMP_ERR_AVG,EMP_ERR_STD],'pac_bound':[BOUND_AVG,BOUND_STD], 'pac_tot_bound':[TOT_BOUND_AVG,TOT_BOUND_STD]}
with open(f"out/uncertainties/epgp_pac.pickle", 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
