import copy
import torch
import numpy as np
import scipy.special
import scipy.spatial
from numpy.random import rand
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class NormalizedConformalRegression():

	def __init__(self, Xc, Yc, trained_model, epsilon=0.05):
		self.Xc = Xc 
		self.Yc = Yc
		self.trained_model = trained_model
		
		self.q = len(Yc) # number of points in the calibration set

		self.epsilon = epsilon


	def get_pred(self, inputs):

		pred, unc = self.trained_model(inputs)
		return pred, unc
		


	def get_calibr_nonconformity_scores(self, y, pred, unc, sorting = True):
		#print(y, pred)
		avg_y = np.mean(y, axis=1)

		ncm = np.abs(pred-avg_y)/unc

		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm


	def get_scores_threshold(self):
		self.calibr_pred, self.calibr_unc = self.get_pred(self.Xc)

		# nonconformity scores on the calibration set
		self.calibr_scores = self.get_calibr_nonconformity_scores(self.Yc, self.calibr_pred, self.calibr_unc)

		Q = (1-self.epsilon)*(1+1/self.q)
		self.tau = np.quantile(self.calibr_scores, Q)

		print("self.tau: ", self.tau)



	def get_cpi(self, inputs):

		pred, unc = self.get_pred(inputs)
		self.get_scores_threshold()

		cpi = np.array([pred-self.tau*unc, pred+self.tau*unc])
		return cpi.T


	def get_coverage_efficiency(self, y_test, test_pred_interval):

		n_points = len(y_test)
		c = 0
		for i in range(n_points):
			if y_test[i] >= test_pred_interval[i,0] and y_test[i] <= test_pred_interval[i,-1]:
				c += 1
		coverage = c/n_points

		efficiency = np.mean(np.abs(test_pred_interval[:,-1]-test_pred_interval[:,0]))

		return coverage, efficiency

