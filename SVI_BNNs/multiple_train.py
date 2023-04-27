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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--n_formulas", default=3, type=int, help="Nb of STL formulas")
parser.add_argument("--likelihood", default='binomial', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--architecture", default='3L', type=str, help="NN architecture")
parser.add_argument("--batch_size", default=100, type=int, help="")
parser.add_argument("--n_epochs", default=2000, type=int, help="Number of training iterations")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--eps", default=0.05, type=float, help="pac epsilon")
parser.add_argument("--n_hidden", default=30, type=int, help="Size of hidden layers")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--device", default="cuda", type=str, help="Choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval)
args = parser.parse_args()

plots_path = os.path.join(plots_path, "SVI_BNNs/")
models_path = os.path.join(models_path, "SVI_BNNs/")

N_TRAINING_RUNS = 5

GEN_ERR_AVG,GEN_ERR_STD = [], []
EMP_ERR_AVG, EMP_ERR_STD = [], []
BOUND_AVG, BOUND_STD = [], []
TOT_BOUND_AVG, TOT_BOUND_STD = [], []

for filepath, train_filename, val_filename, params_list, math_params_list in stat_guar_case_studies:

    if len(params_list)==6:
        args.n_epochs = 100
        args.batch_size = 5000

    print(args)

    print(f"\n=== SVI BNN Training {train_filename} ===")

   
    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== SVI BNN Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    bnn_smmc = BNN_smMC(model_name=filepath, list_param_names=params_list, likelihood=args.likelihood,
        input_size=len(params_list), n_hidden=args.n_hidden, architecture_name=args.architecture)

    gen_err, emp_err, bound, tot_bound = [], [], [], []

    for j in range(N_TRAINING_RUNS):
        i = -1
        print(f"---formula {i}")
        out_filename = f"svi_bnn_{train_filename}_formula_{i}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_hidden={args.n_hidden}_{args.architecture}"
        print(out_filename)
        if args.load:
            bnn_smmc.load(filepath=models_path, filename=out_filename, device=args.device,idx=str(j))
        else:
            bnn_smmc.train(train_data=train_data, pindex = i, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                device=args.device)
            bnn_smmc.save(filepath=models_path, filename=out_filename, training_device=args.device, idx=str(j))


        res_j = bnn_smmc.compute_pac_bound(train_data=train_data, pindex = i, test_data=val_data, device=args.device, epsilon=args.eps)
        
        gen_err.append(res_j[0])
        emp_err.append(res_j[1])
        bound.append(res_j[2].cpu().numpy())
        tot_bound.append(res_j[1]+res_j[2].cpu().numpy())

        post_mean, q1, q2, evaluation_dict = bnn_smmc.evaluate(train_data=train_data, pindex = i, val_data=val_data,
            n_posterior_samples=args.n_posterior_samples, device=args.device)
    
    GEN_ERR_AVG.append(np.mean(gen_err))
    GEN_ERR_STD.append(np.std(gen_err))
    EMP_ERR_AVG.append(np.mean(emp_err))
    EMP_ERR_STD.append(np.std(emp_err))
    BOUND_AVG.append(np.mean(bound))
    BOUND_STD.append(np.std(bound))
    TOT_BOUND_AVG.append(np.mean(tot_bound))
    TOT_BOUND_STD.append(np.std(tot_bound))

    pyro.clear_param_store()

results_dict = {'gen': [GEN_ERR_AVG,GEN_ERR_STD],'emp':[EMP_ERR_AVG,EMP_ERR_STD],'pac_bound':[BOUND_AVG,BOUND_STD], 'pac_tot_bound':[TOT_BOUND_AVG,TOT_BOUND_STD]}
with open(f"out/uncertainties/svibnn_pac.pickle", 'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
