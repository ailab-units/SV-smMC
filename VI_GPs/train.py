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
from bio_settings import *
from posterior_plot_utils import plot_posterior
from VI_GPs.variational_GP import GPmodel
from data_utils import normalize_columns, get_tensor_data, get_tensor_data_w_pindex
from data_utils import normalize_columns, get_bernoulli_data, get_bernoulli_data_w_pindex


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--n_formulas", default=3, type=int, help="Nb of STL formulas")
parser.add_argument("--likelihood", default='bernoulli', type=str, help="Choose 'bernoulli' or 'binomial'")
parser.add_argument("--variational_distribution", default='cholesky', type=str, help="Variational distribution: cholesky, meanfield")
parser.add_argument("--variational_strategy", default='default', type=str, help="Variational strategy: default, unwhitened, batchdecoupled")
parser.add_argument("--load", default=False, type=eval, help="If True load the model else train it")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size")
parser.add_argument("--n_epochs", default=2000, type=int, help="Max number of training iterations")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--n_posterior_samples", default=100, type=int, help="Number of samples from posterior distribution")
parser.add_argument("--device", default="cpu", type=str, help="Choose 'cpu' or 'cuda'")
args = parser.parse_args()

plots_path = os.path.join(plots_path, "VI_GPs/")
models_path = os.path.join(models_path, "VI_GPs/")

for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    print(args)

    print(f"\n=== VI GP Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n===SVI GP Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)

    if args.n_formulas == -1:
        range_vec = np.arange(-1,0)
    else:
        range_vec = np.arange(args.n_formulas)

    for i in range_vec:
        print(f"---formula {i}")
        out_filename = f"vi_gp_{train_filename}_formula_{i}_epochs={args.n_epochs}_lr={args.lr}_batch={args.batch_size}_{args.variational_distribution}_{args.variational_strategy}"

        inducing_points = normalize_columns(get_bernoulli_data(train_data)[0])

        model = GPmodel(inducing_points=inducing_points, variational_distribution=args.variational_distribution,
            variational_strategy=args.variational_strategy, likelihood=args.likelihood)

        if args.load:
            model.load(filepath=models_path, filename=out_filename)

        else:
            
            model.train_gp(train_data=train_data, pindex = i, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                device=args.device)
            model.save(filepath=models_path, filename=out_filename, training_device=args.device)

        
        #_ = model.compute_pac_bounds(train_data=train_data, pindex = i, test_data=val_data, device= 'cpu')

        post_mean, q1, q2, evaluation_dict = model.evaluate(train_data=train_data, val_data=val_data, pindex = i, 
            n_posterior_samples=args.n_posterior_samples, device=args.device)

        '''
        if len(params_list)<=2:

            fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
                test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

            os.makedirs(os.path.dirname(plots_path), exist_ok=True)
            fig.savefig(plots_path+f"{out_filename}.png")
        '''