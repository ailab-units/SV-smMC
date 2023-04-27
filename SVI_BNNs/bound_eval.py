import os
import sys
import pyro
import torch
import random
import argparse
import numpy as np
import pickle5 as pickle

sys.path.append(".")
from settings import *
from SVI_BNNs.bnn import BNN_smMC
from posterior_plot_utils import plot_posterior
from NormalizedCP import *
from CP import *
from data_utils import normalize_columns, get_tensor_data, get_tensor_data_w_pindex

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--n_formulas", default=3, type=int, help="Nb of STL formulas")
parser.add_argument("--likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--architecture", default='3L', type=str, help="NN architecture")
parser.add_argument("--batch_size", default=100, type=int, help="")
parser.add_argument("--n_epochs", default=5000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--eps", default=0.05, type=float, help="pac epsilon")
parser.add_argument("--n_hidden", default=30, type=int, help="Size of hidden layers")
parser.add_argument("--n_posterior_samples", default=500, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--device", default="cuda", type=str, help="Choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval)
args = parser.parse_args()

plots_path = os.path.join(plots_path, "SVI_BNNs/")
models_path = os.path.join(models_path, "SVI_BNNs/")

def pred_function_w_unc(bnn_smmc, n_posterior_samples=500, alpha=0.05):
    def predictor(x):
        with torch.no_grad():
            post_preds, post_mean, post_std = bnn_smmc.forward(torch.FloatTensor(x).to(args.device), n_posterior_samples)

        return post_mean.cpu().numpy(), post_std.cpu().numpy()
    return predictor

def pred_function(bnn_smmc, n_posterior_samples=1000, z=1.96):
    def predictor(x):
        with torch.no_grad():
            _, post_mean, _ = bnn_smmc.forward(torch.FloatTensor(x).to(args.device), n_posterior_samples)
 
        return post_mean.cpu().numpy()
    return predictor

for filepath, train_filename, cal_filename, test_filename, params_list, nb_obs in cp_case_studies:


    print(f"\n=== SVI BNN Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== SVI BNN Calibration {cal_filename} ===")

    with open(os.path.join(data_path, filepath, cal_filename+".pickle"), 'rb') as handle:
        cal_data = pickle.load(handle)

    print(f"\n=== SVI BNN Test {test_filename} ===")

    with open(os.path.join(data_path, filepath, test_filename+".pickle"), 'rb') as handle:
        test_data = pickle.load(handle)
     
    x_train, y_train, n_train_samples, n_train_trials = get_tensor_data_w_pindex(train_data, -1)
    x_cal, y_cal, n_cal_samples, n_cal_trials = get_tensor_data_w_pindex(cal_data, -1)
    x_test, y_test, n_test_samples, n_test_trials = get_tensor_data_w_pindex(test_data, -1)
    
    min_x, max_x, x_train_norm = normalize_columns(x_train, return_minmax=True)
    x_cal_norm = normalize_columns(x_cal, min_x=min_x, max_x=max_x)
    x_test_norm = normalize_columns(x_test, min_x=min_x, max_x=max_x)
    
    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, likelihood=args.likelihood,
        input_size=len(params_list), n_hidden=args.n_hidden, architecture_name=args.architecture)

    predictor_w_unc = pred_function_w_unc(bnn_smmc)
    predictor = pred_function(bnn_smmc)
    
    cp_list, ncp_list, unc_list, smc_list = [],[],[],[]

    for j in range(5):
        i = -1
        print(f"---formula {i}")
        out_filename = f"svi_bnn_{train_filename}_formula_{i}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_hidden={args.n_hidden}_{args.architecture}"
        print(out_filename)
        y_cal = cal_data["labels"][:,:nb_obs]

        smc_mean = np.mean(test_data["labels"],axis=1)
        smc_unc = 1.96*np.std(test_data["labels"],axis=1)/np.sqrt(1000)
        smc_list.append([smc_mean-smc_unc, smc_mean,  smc_mean+smc_unc])

        bnn_smmc.load(filepath=models_path, filename=out_filename, device=args.device,idx=str(j))
        
        cp = ConformalRegression(x_cal_norm, y_cal, predictor)
        ncp = NormalizedConformalRegression(x_cal_norm, y_cal, predictor_w_unc)
        
        cp.get_scores_threshold()
        
        cpi = ncp.get_cpi(x_test_norm)

        post_mean, q1, q2, evaluation_dict = bnn_smmc.evaluate(train_data=train_data, pindex = i, val_data=test_data,
            n_posterior_samples=args.n_posterior_samples, device=args.device)
    
        unc_list.append([q1, q2])
        cp_list.append([post_mean-cp.tau*np.ones(post_mean.shape),post_mean+cp.tau*np.ones(post_mean.shape)])
        ncp_list.append(cpi.T)

      
    results = {'post_mean': post_mean,'cp': (np.mean(cp_list,axis=0),np.std(cp_list,axis=0)), 'ncp':(np.mean(ncp_list,axis=0),np.std(ncp_list,axis=0)), 'bayes_unc': (np.mean(unc_list,axis=0),np.std(unc_list,axis=0)), 'smc_unc': (np.mean(smc_list,axis=0),np.std(smc_list,axis=0))}

    with open(f"out/uncertainties/svibnn_estim_{nb_obs}obs.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    pyro.clear_param_store()  
