import os
import sys
import time
import pyro
import torch
import random
import pickle
import scipy.io
import matplotlib
import numpy as np
from math import pi
import torch.nn as nn
from tqdm import tqdm
from pyro import poutine
import torch.optim as optim
from pyro.nn import PyroModule
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pyro.optim import Adam, SGD
from sklearn import preprocessing
import torch.nn.functional as nnf
from itertools import combinations
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from pyro.distributions import Normal, Binomial, Bernoulli
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

sys.path.append(".")
from data_utils import *
from SVI_BNNs.dnn import DeterministicNetwork
from evaluation_metrics import execution_time, evaluate_posterior_samples

from torch.distributions.multivariate_normal import MultivariateNormal
#from torch.distributions import Normal

softplus = torch.nn.Softplus()


class BNN_smMC(PyroModule):

    def __init__(self, model_name, list_param_names, likelihood, input_size, architecture_name, n_hidden, 
        n_test_points=20):

        # initialize PyroModule
        super(BNN_smMC, self).__init__()
        
        # BayesianNetwork extends PyroModule class
        self.det_network = DeterministicNetwork(input_size=input_size, hidden_size=n_hidden, 
            architecture_name=architecture_name)
        self.name = "bayesian_network"

        self.likelihood = likelihood
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = 1
        self.n_test_points = n_test_points
        self.model_name = model_name
        self.param_name = list_param_names
        self.mre_eps = 0.000001
        self.casestudy_id = self.model_name+''.join(self.param_name)

    def model(self, x_data, y_data):

        priors = {}
    
        # set Gaussian priors on the weights of self.det_network
        for key, value in self.det_network.state_dict().items():
            loc = value #torch.zeros_like(value)
            scale = torch.ones_like(value)/value.size(dim=0)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

        # pyro.random_module places `priors` over the parameters of the nn.Module 
        # self.det_network and returns a distribution, which upon calling 
        # samples a new nn.Module (`lifted_module`)
        lifted_module = pyro.random_module("module", self.det_network, priors)()
    
        # samples are conditionally independent w.r.t. the observed data
        lhat = lifted_module(x_data) # out.shape = (batch_size, num_classes)
        lhat = nnf.sigmoid(lhat)

        if self.likelihood=="binomial":
            pyro.sample("obs", Binomial(total_count=self.n_trials_train, probs=lhat), obs=y_data)

        elif self.likelihood=="bernoulli":
            pyro.sample("obs", Bernoulli(probs=lhat), obs=y_data)

        else:
            raise AttributeError

    def guide(self, x_data, y_data=None):

        dists = {}
        for key, value in self.det_network.state_dict().items():

            # torch.randn_like(x) builds a random tensor whose shape equals x.shape
            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))

            # softplus is a smooth approximation to the ReLU function
            # which constraints the scale tensor to positive values
            distr = Normal(loc=loc, scale=softplus(scale))

            # add key-value pair to the samples dictionary
            dists.update({str(key):distr})
        # define a random module from the dictionary of distributions
        lifted_module = pyro.random_module("module", self.det_network, dists)()

        # compute predictions on `x_data`
        lhat = lifted_module(x_data)
        lhat = nnf.sigmoid(lhat)
        return lhat
    
    def forward(self, inputs, n_samples=100):
        """ Compute predictions on `inputs`. 
        `n_samples` is the number of samples from the posterior distribution.
        If `avg_prediction` is True, it returns the average prediction on 
        `inputs`, otherwise it returns all predictions 
        """

        preds = []
        # take multiple samples
        for i in range(n_samples):    
            pyro.set_rng_seed(i)     
            guide_trace = poutine.trace(self.guide).get_trace(inputs)
            preds.append(guide_trace.nodes['_RETURN']['value'])
        
        t_hats = torch.stack(preds).squeeze()
        t_mean = torch.mean(t_hats, 0)
        t_std = torch.std(t_hats, 0)
        
        return t_hats, t_mean, t_std
    
    def evaluate(self, train_data, pindex, val_data, n_posterior_samples, device="cpu"):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)    

        if self.model_name == 'Poisson':
            raise NotImplementedError

        else:

            x_train = get_binomial_data_w_pindex(train_data, pindex)[0]
            x_val, y_val, n_samples, n_trials = get_tensor_data_w_pindex(val_data, pindex)

            min_x, max_x, _ = normalize_columns(x_train, return_minmax=True)
            x_val = normalize_columns(x_val, min_x=min_x, max_x=max_x) 

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        with torch.no_grad():   

            start = time.time()
            post_samples, post_mean, post_std = self.forward(x_val, n_posterior_samples)
            evaluation_time = execution_time(start=start, end=time.time())
            print(f"Evaluation time = {evaluation_time}")

        post_mean, q1, q2 , evaluation_dict = evaluate_posterior_samples(y_val=y_val,
            post_samples=post_samples, n_samples=n_samples, n_trials=n_trials)

        evaluation_dict.update({"evaluation_time":evaluation_time})
        return post_mean, q1, q2, evaluation_dict


    def compute_pac_bound(self, train_data, test_data, pindex, device, epsilon = 0.05):
        # n = nb of training samples
        # kl_div is the kl divergence between two gaussian distributions

        n_posterior_samples = 100

        x_train, y_train, n_train_samples, n_train_trials = get_tensor_data_w_pindex(train_data, pindex)
        x_test, y_test, n_test_samples, n_test_trials = get_tensor_data_w_pindex(test_data, pindex)

        n = x_train.size(0)
        print('n = ', n)
        print('epsilon = ', epsilon)
        min_x, max_x, x_train = normalize_columns(x_train, return_minmax=True)
        x_test = normalize_columns(x_test, min_x=min_x, max_x=max_x) 

        p_train = torch.mean(y_train,1).to(device)
        p_test = torch.mean(y_test,1).to(device)

        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            # set Gaussian priors on the weights of self.det_network

            prior_dict = self.prior_mean_weights
            
            vv = [prior_dict['model.1.weight'].flatten(), prior_dict['model.1.bias'].flatten(),
                            prior_dict['model.3.weight'].flatten(), prior_dict['model.3.bias'].flatten(),
                            prior_dict['model.5.weight'].flatten(), prior_dict['model.5.bias'].flatten()]
           
            prior_locs = torch.cat((vv[:]),0).to(device)
            
            ss = [torch.ones_like(vv[i])/vv[i].size(0) for i in range(len(vv)) ]

            prior_scales = torch.cat((ss[:]),0).to(device)
            
            post_dict = pyro.get_param_store()
            post_locs = torch.cat((post_dict['model.1.weight_loc'].flatten(), post_dict['model.1.bias_loc'].flatten(),
                            post_dict['model.3.weight_loc'].flatten(), post_dict['model.3.bias_loc'].flatten(),
                            post_dict['model.5.weight_loc'].flatten(), post_dict['model.5.bias_loc'].flatten()),0).to(device)
            post_scales = torch.cat((post_dict['model.1.weight_scale'].flatten(), post_dict['model.1.bias_scale'].flatten(),
                post_dict['model.3.weight_scale'].flatten(), post_dict['model.3.bias_scale'].flatten(),
                post_dict['model.5.weight_scale'].flatten(), post_dict['model.5.bias_scale'].flatten()),0).to(device)
            

            for k in range(len(post_scales)):
                prior_scales[k] = softplus(prior_scales[k])
                post_scales[k] = softplus(post_scales[k])
            
 
            prior_distrib_uni = torch.distributions.Normal(loc=torch.zeros_like(prior_locs), scale=prior_scales)#torch.ones_like(prior_scales)
            post_distrib_uni = torch.distributions.Normal(loc=post_locs, scale=post_scales)
            
            kl_uni = torch.distributions.kl_divergence(post_distrib_uni,prior_distrib_uni)
            kl_div = torch.sum(kl_uni)
            print('kl_div = ', kl_div)

            C = 1 # loss upper boud
            gamma, bound_gamma = self.compute_bound(C, epsilon, n, kl_div)
            
            print('lambda = ', gamma)
            print('bound = ', bound_gamma)

            train_post_samples, train_post_mean, train_post_std = self.forward(x_train, n_posterior_samples)

            emp_err = torch.tensor([torch.mean(torch.abs(p_train-train_post_samples[i]),0) for i in range (len(train_post_samples))])
            emp_err = torch.mean(emp_err,0)
            print('emp_err = ', emp_err)

            test_post_samples, test_post_mean, test_post_std = self.forward(x_test, n_posterior_samples)
            gen_err = torch.tensor([torch.mean(torch.abs(p_test-test_post_samples[i]),0) for i in range (len(test_post_samples))])

            gen_err = torch.mean(gen_err,0)
            print('gen_err = ', gen_err)
        
        return (gen_err, emp_err, bound_gamma, kl_div)
         
         
            
 

    def compute_bound(self, C, epsilon, n, kl_div):

        lamb_star = torch.sqrt( (np.log(1/epsilon)+kl_div)*8*n/(C**2) )

        bound_star = lamb_star*(C**2)/(8*n)+(kl_div+np.log(1/epsilon))/lamb_star

        return lamb_star, bound_star

    def compute_kl_multivar_gauss(self, mu_0, var_0, mu_1, var_1):

        k = len(mu_0)

        sigma_0 = torch.diag(var_0)
        sigma_1 = torch.diag(var_1)

        #inv_sigma_1 = torch.diag(inv_var_1)
        inv_sigma_1 = torch.linalg.inv(sigma_1)

        trace = torch.trace(torch.matmul(inv_sigma_1, sigma_0))
        #trace = torch.sum(var_0/var_1)
        aa = (mu_1-mu_0)[None,:]
        bb = torch.matmul(inv_sigma_1,(mu_1-mu_0)[:,None])
        cc = torch.matmul(aa, bb)
        #dd = torch.prod(var_1)/torch.prod(var_0)
        dd = torch.sum(torch.log(var_1))-torch.sum(torch.log(var_0))
        #kl = (trace-k+cc+torch.log(dd))/2

        kl = (trace-k+cc+dd)/2

        return kl

    def train(self, train_data, pindex, n_epochs, lr, batch_size, device="cpu", idx=''):

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        if self.likelihood=='bernoulli':
            #x_train, y_train, n_samples, n_trials_train = get_bernoulli_data(train_data)
            x_train, y_train, n_samples, n_trials_train = get_bernoulli_data_w_pindex(train_data, pindex)
            
        elif self.likelihood=='binomial':
            #x_train, y_train, n_samples, n_trials_train = get_binomial_data(train_data)
            x_train, y_train, n_samples, n_trials_train = get_binomial_data_w_pindex(train_data, pindex)

        else:
            raise AttributeError

        self.n_trials_train = n_trials_train
        x_train = normalize_columns(x_train)
        y_train = y_train.unsqueeze(1)
        
        dataset = TensorDataset(x_train, y_train) 
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        start = time.time()

        print("\nDeterministic Training:")
        self.det_network.train(train_loader=train_loader, n_trials_train=n_trials_train, epochs=500, 
            lr=0.01, likelihood=self.likelihood, device=device)

        self.prior_mean_weights = self.det_network.state_dict()
        print("\nBayesian Training:")
        # adam_params = {"lr": self.lr, "betas": (0.95, 0.999)}
        adam_params = {"lr": lr}#, "weight_decay":1.}
        optim = Adam(adam_params)
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, loss=elbo)

        loss_history = []
        for j in tqdm(range(n_epochs)):
            loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss += svi.step(x_batch, y_batch)

            loss = loss/len(x_train)

            if (j+1)%50==0:
                print("Epoch ", j+1, "/", n_epochs, " Loss ", loss)
                loss_history.append(loss)

        training_time = execution_time(start=start, end=time.time())

        self.loss_history = loss_history
        self.n_epochs = n_epochs
        self.posterior_weights = pyro.get_param_store()
        print("\nTraining time: ", training_time)
        self.training_time = training_time
        return self, training_time

    def save(self, filepath, filename, training_device, idx=''):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(os.path.join(filepath, filename+idx+"_prior.pickle"), 'wb') as handle:
            pickle.dump(self.prior_mean_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        param_store = pyro.get_param_store()
        print(f"\nlearned params = {param_store}")
        param_store.save(os.path.join(filepath, filename+idx+".pt"))

        file = open(os.path.join(filepath, f"{filename}"+idx+"_training_time.txt"),"w")
        file.writelines(f"{self.training_time} {training_device}")
        file.close()

        if self.n_epochs >= 50:
            fig = plt.figure()
            plt.plot(np.arange(0,self.n_epochs,50), np.array(self.loss_history))
            plt.title("loss")
            plt.xlabel("epochs")
            plt.tight_layout()
            plt.savefig(os.path.join(filepath, filename+"_loss.png"))
            plt.close()          

    def load(self, filepath, filename, device="cpu", idx=''):

        param_store = pyro.get_param_store()
        param_store.load(os.path.join(filepath, filename+idx+".pt"))
        for key, value in param_store.items():
            param_store.replace_param(key, value.to(device), value)

        with open(os.path.join(filepath, filename+idx+"_prior.pickle"), 'rb') as handle:
            self.prior_mean_weights = pickle.load(handle)

        file = open(os.path.join(filepath, f"{filename}"+idx+"_training_time.txt"),"r+")
        training_time = file.read()
        print(f"\nTraining time = {training_time}")
        return training_time