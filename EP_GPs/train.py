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

for filepath, train_filename, val_filename, params_list, math_params_list in case_studies:

    print(args)

    print(f"\n=== EP GP Training {train_filename} ===")

    with open(os.path.join(data_path, filepath, train_filename+".pickle"), 'rb') as handle:
        train_data = pickle.load(handle)

    print(f"\n=== EP GP Validation {val_filename} ===")

    with open(os.path.join(data_path, filepath, val_filename+".pickle"), 'rb') as handle:
        val_data = pickle.load(handle)
        
    if args.n_formulas == -1:
        range_vec = np.arange(-1,0)
    else:
        range_vec = np.arange(0,args.n_formulas)

        for i in range_vec:
        print(f"---formula {i}")
        out_filename = f"ep_gp_{train_filename}_formula_{i}"

        smc = smMC_GPEP()

        #x_train, y_train, n_samples_train, n_trials_train = smc.transform_data(train_data)
        x_train, y_train, n_samples_train, n_trials_train = smc.transform_data_w_pindex(train_data, property_index=i)
        
        #min_x, max_x, x_train = normalize_columns(x_train, return_minmax=True, is_torch=False)

        if args.load:
            smc.load(filepath=models_path, filename=out_filename)

        else:
            # cProfile.run("smc.fit(x_train, y_train, n_trials_train)")
            smc.fit(x_train, y_train, n_trials_train)
            smc.save(filepath=models_path, filename=out_filename)

            
        x_val, y_val, n_samples_val, n_trials_val = smc.transform_data_w_pindex(val_data, property_index=i)
        #x_val = normalize_columns(x_val, min_x=min_x, max_x=max_x, is_torch=False) 
        
        post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_val, y_val=val_data['labels'], 
            n_samples=n_samples_val, n_trials=n_trials_val)
        

        _ = smc.compute_pac_bounds(x_train, y_train, x_val, y_val)


        #post_mean, q1, q2, evaluation_dict = smc.eval_gp(x_train=x_train, x_val=x_val, y_val=y_val, 
            #n_samples=n_samples_val, n_trials=n_trials_val)

    '''
    if len(params_list)<=2:

        fig = plot_posterior(params_list=params_list, math_params_list=math_params_list, train_data=train_data,
            test_data=val_data, val_data=val_data, post_mean=post_mean, q1=q1, q2=q2)

        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        fig.savefig(plots_path+f"{out_filename}.png")
    '''
