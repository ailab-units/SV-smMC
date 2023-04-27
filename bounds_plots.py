import os
import sys
import numpy as np
from settings import *
import pickle5 as pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 40})
from data_utils import normalize_columns, get_tensor_data, get_tensor_data_w_pindex


chern = [0.4294694083467376,0.2716203031481239,0.19206455826398416,0.13581015157406195,0.09603227913199208,0.060736146190830516]#0.04294694083467376

SMCs = ["EP-GP", "SVI-GP", "SVI-BNN"]
col = ['g','b','r','grey']

CHERN = False
LW = 3
cp_col = 'turquoise' 
ncp_col = 'orange' 
chern_col = 'purple'
c = 0
for filepath, _, _, test_filename, params_list, nb_obs in cp_case_studies:


	print(f"\n=== SVI BNN Test {test_filename} ===")
	with open(os.path.join(data_path, filepath, test_filename+".pickle"), 'rb') as handle:
		test_data = pickle.load(handle)
	 
	x_test, y_test, n_test_samples, n_test_trials = get_tensor_data_w_pindex(test_data, -1)
	
	with open(f"out/uncertainties/svibnn_estim_{nb_obs}obs.pickle", 'rb') as handle:
		svibnn = pickle.load(handle)
	with open(f"out/uncertainties/svigp_estim_{nb_obs}obs.pickle", 'rb') as handle:
		svigp = pickle.load(handle)
	with open(f"out/uncertainties/epgp_estim_{nb_obs}obs.pickle", 'rb') as handle:
		epgp = pickle.load(handle)

	subset_idx = np.arange(0,1000,20)
	fig,ax = plt.subplots(1,3, figsize=(36,12))
	ax[0].grid(True)
	ax[0].plot(x_test[:,0], epgp['post_mean'], color=col[0],linewidth=LW)
	
	ax[0].plot(x_test[:,0], [max(0,u) for u in epgp['cp'][0][0]], color=cp_col,label='icp', linewidth=LW, linestyle='--')
	ax[0].plot(x_test[:,0], epgp['cp'][0][1], color=cp_col, linewidth=LW, linestyle='--')
	ax[0].plot(x_test[:,0], [max(0,u) for u in epgp['ncp'][0][0]], color=ncp_col,label='nicp', linewidth=LW, linestyle='--')
	ax[0].plot(x_test[:,0], epgp['ncp'][0][1], color=ncp_col, linewidth=LW, linestyle='--')
	if CHERN:
		ax[0].plot(x_test[:,0], [max(0,u-chern[c]) for u in epgp['ncp'][0][0]], color=chern_col,label='chernoff', linewidth=LW, linestyle='--')
		ax[0].plot(x_test[:,0], epgp['ncp'][0][1]+chern[c], color=chern_col, linewidth=LW, linestyle='--')

	ax[0].plot(x_test[:,0], [max(0,u) for u in epgp['bayes_unc'][0][0]], color=col[0],label='unc',linewidth=LW,linestyle='--')
	ax[0].plot(x_test[:,0], epgp['bayes_unc'][0][1], color=col[0],linewidth=LW,linestyle='--')
	ax[0].errorbar(x_test[subset_idx,0], y=epgp['smc_unc'][0][1][subset_idx], yerr=[epgp['smc_unc'][0][1][subset_idx]-epgp['smc_unc'][0][0][subset_idx],epgp['smc_unc'][0][2][subset_idx]-epgp['smc_unc'][0][1][subset_idx]],  color = col[-1], fmt='o', capsize = 0, label='SMC')

	ax[0].set_title(SMCs[0])
	ax[0].legend(prop={'size': 36})

	ax[1].grid(True)
	ax[1].plot(x_test[:,0], svigp['post_mean'], color=col[1],linewidth=LW)
	
	ax[1].plot(x_test[:,0], [max(0,u) for u in svigp['cp'][0][0]], color=cp_col,label='icp', linewidth=LW, linestyle='--')
	ax[1].plot(x_test[:,0], svigp['cp'][0][1], color=cp_col, linewidth=LW, linestyle='--')
	ax[1].plot(x_test[:,0], [max(0,u) for u in svigp['ncp'][0][0]], color=ncp_col,label='nicp', linewidth=LW, linestyle='--')
	ax[1].plot(x_test[:,0], svigp['ncp'][0][1], color=ncp_col, linewidth=LW, linestyle='--')
	if CHERN:
		ax[1].plot(x_test[:,0], [max(0,u-chern[c]) for u in svigp['ncp'][0][0]], color=chern_col,label='chernoff', linewidth=LW, linestyle='--')
		ax[1].plot(x_test[:,0], svigp['ncp'][0][1]+chern[c], color=chern_col, linewidth=LW, linestyle='--')
	
	ax[1].plot(x_test[:,0], [max(0,u) for u in svigp['bayes_unc'][0][0]], color=col[1],label='unc',linewidth=LW,linestyle='--')
	ax[1].plot(x_test[:,0], svigp['bayes_unc'][0][1], color=col[1],linewidth=LW,linestyle='--')
	ax[1].errorbar(x_test[subset_idx,0], y=epgp['smc_unc'][0][1][subset_idx], yerr=[epgp['smc_unc'][0][1][subset_idx]-epgp['smc_unc'][0][0][subset_idx],epgp['smc_unc'][0][2][subset_idx]-epgp['smc_unc'][0][1][subset_idx]],  color = col[-1], fmt='o', capsize = 0, label='SMC')

	ax[1].set_title(SMCs[1])
	ax[1].legend(prop={'size': 36})

	ax[2].grid(True)
	ax[2].plot(x_test[:,0], svibnn['post_mean'], color=col[2],linewidth=LW)
	
	ax[2].plot(x_test[:,0], [max(0,u) for u in svibnn['cp'][0][0]], color=cp_col,label='icp', linewidth=LW, linestyle='--')
	ax[2].plot(x_test[:,0], svibnn['cp'][0][1], color=cp_col, linewidth=LW, linestyle='--')
	ax[2].plot(x_test[:,0], [max(0,u) for u in svibnn['ncp'][0][0]], color=ncp_col,label='nicp', linewidth=LW, linestyle='--')
	ax[2].plot(x_test[:,0], svibnn['ncp'][0][1], color=ncp_col, linewidth=LW, linestyle='--')
	if CHERN:
		ax[2].plot(x_test[:,0], [max(0,u-chern[c]) for u in svibnn['ncp'][0][0]], color=chern_col,label='chernoff', linewidth=LW, linestyle='--')
		ax[2].plot(x_test[:,0], svibnn['ncp'][0][1]+chern[c], color=chern_col, linewidth=LW, linestyle='--')
	
	ax[2].plot(x_test[:,0], [max(0,u) for u in svibnn['bayes_unc'][0][0]], color=col[2],label='unc',linewidth=LW,linestyle='--')
	ax[2].plot(x_test[:,0], svibnn['bayes_unc'][0][1], color=col[2],linewidth=LW,linestyle='--')
	ax[2].errorbar(x_test[subset_idx,0], y=epgp['smc_unc'][0][1][subset_idx], yerr=[epgp['smc_unc'][0][1][subset_idx]-epgp['smc_unc'][0][0][subset_idx],epgp['smc_unc'][0][2][subset_idx]-epgp['smc_unc'][0][1][subset_idx]],  color = col[-1], fmt='o', capsize = 0, label='SMC')

	ax[2].set_title(SMCs[2])
	ax[2].legend(prop={'size': 36})
	plt.suptitle(r'$M_t$={}'.format(nb_obs))
	plt.tight_layout()
	if CHERN:
		plt.savefig(f'out/plots/compare_unc_{nb_obs}obs_n=500_w_chernoff.png')
	else:
		plt.savefig(f'out/plots/compare_unc_{nb_obs}obs_n=500.png')
	
	plt.close()

	c+=1