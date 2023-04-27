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
from NormalizedCP import *
from CP import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_formulas", default=3, type=int, help="Nb of STL formulas")
args = parser.parse_args()

plots_path = os.path.join(plots_path, "EP_GPs/")
models_path = os.path.join(models_path, "EP_GPs/")

TAUS_AVG, TAUS_STD = [],[]
NORM_TAUS_AVG, NORM_TAUS_STD = [],[]
UNC_AVG, UNC_STD = [],[]
SMC_AVG, SMC_STD = [],[]

for filepath, train_filename, cal_filename, test_filename, params_list, nb_obs in cp_case_studies:

    print(args)

    print(f"\n=== EP GP Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== EP GP Calibration {cal_filename} ===")

    with open(os.path.join(data_path, filepath, cal_filename+".pickle"), 'rb') as handle:
        cal_data = pickle.load(handle)

    print(f"\n=== EP GP Test {test_filename} ===")

    with open(os.path.join(data_path, filepath, test_filename+".pickle"), 'rb') as handle:
        test_data = pickle.load(handle)
    
    cp_list, ncp_list, unc_list, smc_list = [],[],[],[]

    for j in range(5):
        i = -1
        print(f"---formula {i}")
        out_filename = f"ep_gp_{train_filename}_formula_{i}"

        smc = smMC_GPEP()

        x_train, y_train, n_samples_train, n_trials_train = smc.transform_data_w_pindex(train_data, property_index=i)
        

        smc.load(filepath=models_path, filename=out_filename, idx=str(j))
            
        x_cal, p_cal, n_samples_cal, n_trials_cal = smc.transform_data_w_pindex(cal_data, property_index=i)
        y_cal = cal_data["labels"][:,:nb_obs]
        

        x_test, y_test, n_samples_test, n_trials_test = smc.transform_data_w_pindex(test_data, property_index=i)

        smc_mean = np.mean(test_data["labels"],axis=1)
        smc_unc = 1.96*np.std(test_data["labels"],axis=1)/np.sqrt(1000)
        smc_list.append([smc_mean-smc_unc, smc_mean,  smc_mean+smc_unc])
        
        post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_test, y_val=test_data['labels'], 
            n_samples=n_samples_test, n_trials=n_trials_test)

        

        predictor = lambda x: smc.avg_predictions(x_train, x)
        predictor_w_unc = lambda x: smc.avg_predictions_w_unc(x_train, x)
        
        cp = ConformalRegression(x_cal, y_cal, predictor)
        cp.get_scores_threshold()
        
        ncp = NormalizedConformalRegression(x_cal, y_cal, predictor_w_unc)
        cpi = ncp.get_cpi(x_test)

    
        unc_list.append([q1, q2])
        cp_list.append([post_mean-cp.tau*np.ones(post_mean.shape),post_mean+cp.tau*np.ones(post_mean.shape)])
        ncp_list.append(cpi.T)
    
    
    results = {'post_mean': post_mean, 'cp': (np.mean(cp_list,axis=0),np.std(cp_list,axis=0)), 'ncp':(np.mean(ncp_list,axis=0),np.std(ncp_list,axis=0)), 'bayes_unc': (np.mean(unc_list,axis=0),np.std(unc_list,axis=0)), 'smc_unc': (np.mean(smc_list,axis=0),np.std(smc_list,axis=0))}

    with open(f"out/uncertainties/epgp_estim_{nb_obs}obs.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)    