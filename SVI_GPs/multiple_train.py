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
parser.add_argument("--device", default="cuda", type=str, help="Choose 'cpu' or 'cuda'")
args = parser.parse_args()

plots_path = os.path.join(plots_path, "SVI_GPs/")
models_path = os.path.join(models_path, "SVI_GPs/")

N_TRAINING_RUNS = 5

GEN_ERR_AVG,GEN_ERR_STD = [], []
EMP_ERR_AVG, EMP_ERR_STD = [], []
BOUND_AVG, BOUND_STD = [], []
TOT_BOUND_AVG, TOT_BOUND_STD = [], []


for filepath, train_filename, val_filename, params_list, math_params_list in stat_guar_case_studies:

    if len(params_list)==6:
        args.n_epochs = 1000
        args.batch_size = 5000

    print(args)

    print(f"\n=== SVI GP Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== SVI GP Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    gen_err, emp_err, bound, tot_bound = [], [], [], []
        
    for j in range(N_TRAINING_RUNS):
        i = -1
        print(f"---formula {i}")
        out_filename = f"svi_gp_{train_filename}_formula_{i}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_{args.variational_distribution}_{args.variational_strategy}"

        inducing_points = normalize_columns(get_tensor_data_w_pindex(train_data, property_index=i)[0])

        model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
            variational_strategy=args.variational_strategy, likelihood=args.likelihood)

        if args.load:
            model.load(filepath=models_path, filename=out_filename, idx=str(j))

        else:
            
            model.train_gp(train_data=train_data, pindex = i, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                device=args.device)
            model.save(filepath=models_path, filename=out_filename, training_device=args.device, idx=str(j))

        
        res_j = model.compute_pac_bounds(train_data=train_data, pindex = i, test_data=val_data, device= 'cpu')

        gen_err.append(res_j[0])
        emp_err.append(res_j[1])
        bound.append(res_j[2])
        tot_bound.append(res_j[1]+res_j[2])

        post_mean, q1, q2, evaluation_dict = model.evaluate(train_data=train_data, val_data=val_data, pindex = i, 
            n_posterior_samples=args.n_posterior_samples, device=args.device)


    GEN_ERR_AVG.append(np.mean(gen_err))
    GEN_ERR_STD.append(np.std(gen_err))
    EMP_ERR_AVG.append(np.mean(emp_err))
    EMP_ERR_STD.append(np.std(emp_err))
    BOUND_AVG.append(np.mean(bound))
    BOUND_STD.append(np.std(bound))
    TOT_BOUND_AVG.append(np.mean(tot_bound))
    TOT_BOUND_STD.append(np.std(tot_bound))


results_dict = {'gen': [GEN_ERR_AVG,GEN_ERR_STD],'emp':[EMP_ERR_AVG,EMP_ERR_STD],'pac_bound':[BOUND_AVG,BOUND_STD], 'pac_tot_bound':[TOT_BOUND_AVG,TOT_BOUND_STD]}
with open(f"out/uncertainties/svigp_pac.pickle", 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
