import os
import sys
import torch
import random
import gpytorch
import argparse
import numpy as np
from math import sqrt
import pickle5 as pickle
import matplotlib.pyplot as plt

sys.path.append(".")
from settings import *
from posterior_plot_utils import plot_posterior
from SVI_GPs.variational_GP import GPmodel
from data_utils import normalize_columns, get_tensor_data, get_tensor_data_w_pindex
from NormalizedCP import *
from CP import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--n_formulas", default=3, type=int, help="Nb of STL formulas")
parser.add_argument("--likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution: cholesky, meanfield")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy: default, unwhitened, batchdecoupled")
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--n_epochs", default=2000, type=int, help="Max number of training iterations")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--device", default="cpu", type=str, help="Choose 'cpu' or 'cuda'")
args = parser.parse_args()

plots_path = os.path.join(plots_path, "SVI_GPs/")
models_path = os.path.join(models_path, "SVI_GPs/")

TAUS_AVG, TAUS_STD = [], []
NORM_TAUS_AVG, NORM_TAUS_STD = [], []
SMC_AVG, SMC_STD = [],[]
UNC_AVG, UNC_STD = [], []
for filepath, train_filename, cal_filename, test_filename, params_list, nb_obs in cp_case_studies:

    if len(params_list)==6:
        args.n_epochs = 1000
        args.batch_size = 5000

    print(args)

    print(f"\n=== SVI GP Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== SVI GP Calibration {cal_filename} ===")

    with open(os.path.join(data_path, filepath, cal_filename+".pickle"), 'rb') as handle:
        cal_data = pickle.load(handle)

    print(f"\n=== SVI GP Test {test_filename} ===")

    with open(os.path.join(data_path, filepath, test_filename+".pickle"), 'rb') as handle:
        test_data = pickle.load(handle)

    x_train, y_train, n_train_samples, n_train_trials = get_tensor_data_w_pindex(train_data, -1)
    x_test, y_test, n_test_samples, n_test_trials = get_tensor_data_w_pindex(test_data, -1)
    x_cal, y_cal, n_cal_samples, n_cal_trials = get_tensor_data_w_pindex(cal_data, -1)
    

    min_x, max_x, x_train_norm = normalize_columns(x_train, return_minmax=True)
    x_test_norm = normalize_columns(x_test, min_x=min_x, max_x=max_x)
    x_cal_norm = normalize_columns(x_cal, min_x=min_x, max_x=max_x)
    
    cp_wid, ncp_wid, unc_wid, smc_wid = [],[],[],[]

    for j in range(5):
        i = -1
        print(f"---formula {i}")
        out_filename = f"svi_gp_{train_filename}_formula_{i}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_{args.variational_distribution}_{args.variational_strategy}"
        y_cal = cal_data["labels"][:,:nb_obs]
        
        smc_unc = 1.96*np.std(test_data["labels"],axis=1)/np.sqrt(1000)
        smc_wid.append(np.mean(2*smc_unc))# media su tutto il test set

        inducing_points = normalize_columns(get_tensor_data_w_pindex(train_data, property_index=i)[0])

        model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
            variational_strategy=args.variational_strategy, likelihood=args.likelihood)

        model.load(filepath=models_path, filename=out_filename, idx=str(j))
        
        _, _, _, evaluation_dict = model.evaluate(train_data=train_data, val_data=test_data, pindex = i, 
            n_posterior_samples=args.n_posterior_samples, device=args.device)
        
        unc_wid.append(evaluation_dict["avg_uncertainty_area"])

        cp = ConformalRegression(x_cal_norm, y_cal, model.avg_predictions)
        cp.get_scores_threshold()
        
        ncp = NormalizedConformalRegression(x_cal_norm, y_cal, model.avg_predictions_w_unc)
        cpi = ncp.get_cpi(x_test_norm)

        cp_wid.append(2*cp.tau)
        ncp_wid.append(np.mean(np.abs(cpi[:,1]-cpi[:,0])))

    results = {'cp': cp_wid, 'ncp':ncp_wid, 'bayes_unc': unc_wid, 'smc_unc': smc_wid}

    with open(f"out/uncertainties/svigp_widths_{nb_obs}obs.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    NORM_TAUS_AVG.append(np.mean(ncp_wid))
    NORM_TAUS_STD.append(np.std(ncp_wid))
    TAUS_AVG.append(np.mean(cp_wid))
    TAUS_STD.append(np.std(cp_wid))
    UNC_AVG.append(np.mean(unc_wid))
    UNC_STD.append(np.std(unc_wid))
    SMC_AVG.append(np.mean(smc_wid))
    SMC_STD.append(np.std(smc_wid))

results_dict = {'CP':[TAUS_AVG,TAUS_STD], 'NCP':[NORM_TAUS_AVG,NORM_TAUS_STD],'BAYES_UNC':[UNC_AVG,UNC_STD],'SMC':[SMC_AVG,SMC_STD]}
with open(f"out/uncertainties/svigp_widths.pickle", 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    

