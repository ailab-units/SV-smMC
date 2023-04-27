import os
import sys
import time
import torch
import random
import gpytorch
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from itertools import product
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.functions import log_normal_cdf
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy, BatchDecoupledVariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution, CholeskyVariationalDistribution

sys.path.append(".")
from SVI_GPs.binomial_likelihood import BinomialLikelihood
from SVI_GPs.bernoulli_likelihood import BernoulliLikelihood
from evaluation_metrics import execution_time, intervals_intersection, evaluate_posterior_samples
from data_utils import normalize_columns, Poisson_observations, get_tensor_data, get_bernoulli_data, get_binomial_data
from data_utils import get_tensor_data_w_pindex, get_bernoulli_data_w_pindex, get_binomial_data_w_pindex



class GPmodel(ApproximateGP):

    def __init__(self, inducing_points, likelihood='binomial', variational_distribution='cholesky', 
        variational_strategy='default', learn_inducing_locations=False):

        learn_inducing_locations = True if len(inducing_points) > 1000 else False

        if len(inducing_points)>1000:
            idxs = np.linspace(0,len(inducing_points)-1,1000).astype(int)
            inducing_points = inducing_points[idxs]

        if variational_distribution=='cholesky':
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        elif variational_distribution=='meanfield':
            variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
        else:
            raise NotImplementedError

        if variational_strategy=='default':
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, 
                                                                learn_inducing_locations=learn_inducing_locations)
        elif variational_strategy=='unwhitened':
            variational_strategy = UnwhitenedVariationalStrategy(self, inducing_points, variational_distribution, 
                                                                learn_inducing_locations=learn_inducing_locations)
        elif variational_strategy=='batchdecoupled':
            variational_strategy = BatchDecoupledVariationalStrategy(self, inducing_points, variational_distribution, 
                                                                learn_inducing_locations=learn_inducing_locations)
        else:
            raise NotImplementedError

        super(GPmodel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = likelihood

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def load(self, filepath, filename, idx=''):
        state_dict = torch.load(os.path.join(filepath, filename+idx+".pth"))
        self.load_state_dict(state_dict)

        file = open(os.path.join(filepath, f"{filename}"+idx+"_training_time.txt"),"r+")
        training_time = file.read()
        print(f"\nTraining time = {training_time}")
        return training_time

    def save(self, filepath, filename, training_device, idx=''):
        print(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), os.path.join(filepath, filename+idx+".pth"))

        file = open(os.path.join(filepath, f"{filename}"+idx+"_training_time.txt"),"w")
        file.writelines(f"{self.training_time} {training_device}")
        file.close()

        if self.n_epochs >= 50:

            fig,ax = plt.subplots()
            x = np.linspace(0,self.n_epochs,len(self.loss_history))
            ax.plot(x, np.array(self.loss_history), color="red")
            ax.set_ylabel("loss",color="red")

            ax2=ax.twinx()
            n_avg_variation_pts = len(self.avg_variation_history)
            x = np.linspace(1,self.n_epochs,len(self.avg_variation_history))
            ax2.plot(x, np.array(self.avg_variation_history), color="blue")
            ax2.set_ylabel("avg variation",color="blue")

            plt.xlabel("epochs")
            plt.tight_layout()
            plt.savefig(os.path.join(filepath, filename+".png"))
            plt.close()   

    def train_gp(self, train_data, pindex, n_epochs, lr, batch_size, device="cpu"):
        #random.seed(0)
        #np.random.seed(0)
        #torch.manual_seed(0)

        if self.likelihood=='bernoulli':
            #x_train, y_train, n_samples, n_trials = get_bernoulli_data(train_data)
            x_train, y_train, n_samples, n_trials = get_bernoulli_data_w_pindex(train_data, pindex)
            likelihood = BernoulliLikelihood()

        elif self.likelihood=='binomial':
            #x_train, y_train, n_samples, n_trials = get_binomial_data(train_data)
            x_train, y_train, n_samples, n_trials = get_binomial_data_w_pindex(train_data, pindex)
            likelihood = BinomialLikelihood()

        else:
            raise AttributeError


        self.train()
        likelihood.train()
        likelihood.n_trials = n_trials

        x_train = normalize_columns(x_train)
        dataset = TensorDataset(x_train, y_train) 
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        trasp_x_train = torch.transpose(x_train, 0, 1)
        x_val = [torch.linspace(x_train_par.min(), x_train_par.max(), steps=1000) for x_train_par in trasp_x_train]
        x_val = torch.stack(x_val).to(device)
        x_val = torch.transpose(x_val, 0, 1)

        self.to(device)
        x_train = x_train.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elbo = gpytorch.mlls.VariationalELBO(likelihood, self, num_data=len(x_train))


        loss_history = []
        avg_variation_history = []
        start = time.time()

        i = 0
        avg_variation = torch.tensor(1)
        while avg_variation>10e-6 and i<n_epochs:

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                output = self(x_batch) # var_strategy (x_batch)
                loss = -elbo(output, y_batch)   
                loss.backward()
                optimizer.step()

            val_out = self.posterior_predictive(x_train=x_train, x_test=x_val, n_posterior_samples=1)[0]

            if i>0:
                avg_variation = torch.mean(torch.norm(val_out-prev_val_out, p=float('inf')))

            prev_val_out = val_out

            if i % 10 == 0:
                avg_variation_history.append(avg_variation.detach().cpu().numpy())
                loss_history.append(loss.detach().cpu().numpy())

                print(f"Epoch {i}/{n_epochs} - Loss: {loss} - Avg variation: {avg_variation}")

            i += 1

        learned_lenghtscale = self.covar_module._modules['base_kernel']._parameters['raw_lengthscale']
        print("\nlearned_lenghtscale =", learned_lenghtscale.item())

        training_time = execution_time(start=start, end=time.time())
        print("\nTraining time =", training_time)

        print("\nModel params:", self.state_dict().keys())
        self.training_time = training_time

        self.loss_history = loss_history
        self.avg_variation_history = avg_variation_history
        self.n_epochs = i

    def posterior_predictive(self, x_train, x_test, n_posterior_samples):
        min_x, max_x, _ = normalize_columns(x_train, return_minmax=True)
        normalized_x = normalize_columns(x_test, min_x=min_x, max_x=max_x)

        posterior = self(normalized_x)
        post_samples = posterior.sample(sample_shape=torch.Size((n_posterior_samples,)))
        post_samples = [[torch.exp(log_normal_cdf(post_samples[j, i]))  for i in range(len(x_test))] \
            for j in range(n_posterior_samples)]
        post_samples = torch.tensor(post_samples)
        return post_samples

    def compute_pac_bounds(self, train_data, test_data, pindex, device, epsilon = 0.05):
        #n_posterior_samples = 100   
        self.to(device)
        x_train, y_train, n_train_samples, n_train_trials = get_tensor_data_w_pindex(train_data, pindex)
        x_test, y_test, n_test_samples, n_test_trials = get_tensor_data_w_pindex(test_data, pindex)
        
        n = x_train.size(0)
        print('n = ', n)
        print('epsilon = ', epsilon)

        print(device)

        min_x, max_x, x_train_norm = normalize_columns(x_train, return_minmax=True)
        x_test_norm = normalize_columns(x_test, min_x=min_x, max_x=max_x)
        
        p_train = torch.mean(y_train,1).to(device)
        p_test = torch.mean(y_test,1).to(device)

        x_train = x_train.to(device)
        x_train_norm = x_train_norm.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        x_test_norm = x_test_norm.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            inducings = x_train_norm

            N = len(inducings)
            if N > 500:
                inducings = inducings[0:2:N]
            print(len(inducings))
            print(self.variational_strategy.inducing_points.shape)
            posterior = self.forward(inducings)
            prior_mean_module = gpytorch.means.ConstantMean()
            prior_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            LEN = self.covar_module._modules['base_kernel']._parameters['raw_lengthscale']
            print('prior lengthscale = ', prior_covar_module._modules['base_kernel']._parameters['raw_lengthscale'])
            print('lengthscale = ', LEN)
            #prior_covar_module._modules['base_kernel']._parameters['raw_lengthscale'] = LEN
            prior_covar_module._modules['base_kernel']._parameters['raw_lengthscale'] = torch.tensor([-0.8])
            
            prior_mean = prior_mean_module(inducings)
            prior_covar = prior_covar_module(inducings)

            prior = gpytorch.distributions.MultivariateNormal(prior_mean, prior_covar)
            
            kl_div = torch.distributions.kl_divergence(posterior,prior)
            print('kl_div = ', kl_div)
            
            C = 1 # loss upper bound
        
            gamma, bound_gamma = self.compute_bound(C, epsilon, n, kl_div)
            
            print('lambda = ', gamma)
            print('bound = ', bound_gamma)

            
            test_post_mean, _, _, _ = self.evaluate_analytic_posterior(x_train=x_train, x_val=x_test, 
                y_val=y_test, n_samples=n_test_samples, n_trials=n_test_trials, print_results=False)
            #print('test_post_mean = ', test_post_mean)
            gen_err = torch.mean(torch.abs(p_test-test_post_mean))
            print('gen_err = ', gen_err)
            
            train_post_mean, _, _, _ = self.evaluate_analytic_posterior(x_train=x_train, x_val=x_train, 
                y_val=y_train, n_samples=n_train_samples, n_trials=n_train_trials, print_results=False)
            #print('train_post_mean = ', train_post_mean)
            emp_err = torch.mean(torch.abs(p_train-train_post_mean))
            print('emp_err = ', emp_err)

        return (gen_err, emp_err, bound_gamma, kl_div)

        
    def compute_bound(self, C, epsilon, n, kl_div):

        lamb_star = torch.sqrt( (np.log(1/epsilon)+kl_div)*8*n/(C**2) )

        bound_star = lamb_star*(C**2)/(8*n)+(kl_div+np.log(1/epsilon))/lamb_star

        return lamb_star, bound_star

    def avg_predictions(self, x_norm):
        with torch.no_grad():
            x = torch.tensor(x_norm)
            post = self(x)
            mean = post.mean.cpu().detach().numpy()
            variance = post.variance.cpu().detach().numpy()
            post_mean = norm.cdf(mean / np.sqrt(1 + variance))

        return post_mean

    def avg_predictions_w_unc(self, x_norm, z=1.96):
        with torch.no_grad():
            x = torch.tensor(x_norm)
            post = self(x)
            mean = post.mean.cpu().detach().numpy()
            variance = post.variance.cpu().detach().numpy()
            post_mean = norm.cdf(mean / np.sqrt(1 + variance))
            quantiles_interval = [mean - z * np.sqrt(variance), mean + z * np.sqrt(variance)]
            
            bounds = norm.cdf(np.tile(1 / np.sqrt(1 + variance), (2, 1)) * quantiles_interval)
            q1, q2 = bounds[0, :], bounds[1, :]
        return post_mean, (q2-q1)/(2*z)

    def evaluate_analytic_posterior(self, x_train, x_val, y_val, n_samples, n_trials, z=1.96, print_results=True):

        if y_val.shape != (n_samples, n_trials):
            raise ValueError("y_val should be bernoulli trials")

        if type(y_val)==torch.Tensor:
            y_val = y_val.cpu().detach().numpy()

        satisfaction_prob = y_val.mean(1).flatten()
        assert satisfaction_prob.min()>=0
        assert satisfaction_prob.max()<=1

        min_x, max_x, _ = normalize_columns(x_train, return_minmax=True)
        normalized_x = normalize_columns(x_val, min_x=min_x, max_x=max_x)
        posterior = self(normalized_x)

        mean = posterior.mean.cpu().detach().numpy()
        variance = posterior.variance.cpu().detach().numpy()

        # post_mean = norm.cdf(mean)
        post_mean = norm.cdf(mean / np.sqrt(1 + variance))
        quantiles_interval = [mean - z * np.sqrt(variance), mean + z * np.sqrt(variance)]
        bounds = norm.cdf(np.tile(1 / np.sqrt(1 + variance), (2, 1)) * quantiles_interval)
        q1, q2 = bounds[0, :], bounds[1, :]

        assert satisfaction_prob.shape == post_mean.shape
        assert satisfaction_prob.shape == q1.shape

        sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val]
        val_std = np.sqrt(sample_variance).flatten()
        validation_ci = (satisfaction_prob-(z*val_std)/np.sqrt(n_trials),satisfaction_prob+(z*val_std)/np.sqrt(n_trials))
        
        q1[q1<10e-6] = 0
        estimated_ci = (q1, q2)

        non_empty_intersections = np.sum(intervals_intersection(validation_ci,estimated_ci))
        val_accuracy = 100*non_empty_intersections/n_samples
        assert val_accuracy <= 100

        val_dist = np.abs(satisfaction_prob-post_mean)
        mse = np.mean(val_dist**2)
        # mre = np.mean(val_dist/satisfaction_prob+0.000001)

        ci_uncertainty_area = q2-q1
        avg_uncertainty_area = np.mean(ci_uncertainty_area)
        if print_results:
            print(f"Mean squared error: {mse}")
            # print(f"Mean relative error: {mre}")
            print(f"Validation accuracy: {val_accuracy} %")
            print(f"Average uncertainty area:  {avg_uncertainty_area}\n")

        evaluation_dict = {"val_accuracy":val_accuracy, "mse":mse, 
            "uncertainty_area":ci_uncertainty_area, "avg_uncertainty_area":avg_uncertainty_area}
        return post_mean, q1, q2, evaluation_dict

    def evaluate(self, train_data, pindex, n_posterior_samples, val_data=None, device="cpu"):

        #random.seed(0)
        #np.random.seed(0)
        #torch.manual_seed(0)

        #x_train = get_binomial_data(train_data)[0]
        #x_val, y_val, n_samples, n_trials = get_tensor_data(val_data)
        x_train = get_binomial_data_w_pindex(train_data, pindex)[0]
        x_val, y_val, n_samples, n_trials = get_tensor_data_w_pindex(val_data, pindex)

        self.eval()    
        self.to(device)
        x_train = x_train.to(device)
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        start = time.time()
        with torch.no_grad():

            ### posterior samples

            # post_samples = self.posterior_predictive(x_train=x_train, x_test=x_val, 
            #     n_posterior_samples=n_posterior_samples)
            # post_mean, q1, q2, evaluation_dict = evaluate_posterior_samples(y_val=y_val, post_samples=post_samples, 
            #     n_samples=n_samples, n_trials=n_trials)

            ### analytic posterior

            post_mean, q1, q2, evaluation_dict = self.evaluate_analytic_posterior(x_train=x_train, x_val=x_val, 
                y_val=y_val, n_samples=n_samples, n_trials=n_trials)
        
        evaluation_time = execution_time(start=start, end=time.time())
        print(f"Evaluation time = {evaluation_time}")

        evaluation_dict.update({"evaluation_time":evaluation_time})
        return post_mean, q1, q2, evaluation_dict
