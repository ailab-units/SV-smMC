import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 36})

from results import *
nsamples = 3*3
Z = 1.96/np.sqrt(nsamples)
test_uncertainties = [0.01554758334447687,0.02071578114636744,0.022368671678692956
,0.01743578856993807,0.007455428294685914,0.010446340114250953
,0.02431320577027406]

#SMCs = ["SVI-GP", "SVI-BNN", "EP-GP"]
col = ['b','r','g','k']

alp = 0.1

svigp_acc_mean = np.mean(np.mean(SVI_GP["ACC"],axis=1),axis=1)
svigp_acc_std = np.std(np.std(SVI_GP["ACC"],axis=1),axis=1)
svibnn_acc_mean = np.mean(np.mean(SVI_BNN["ACC"],axis=1),axis=1)
svibnn_acc_std = np.std(np.std(SVI_BNN["ACC"],axis=1),axis=1)
epgp_acc_mean = np.mean(np.mean(EP_GP["ACC"],axis=1),axis=1)
epgp_acc_std = np.std(np.std(EP_GP["ACC"],axis=1),axis=1)

fig,ax = plt.subplots(2,2, figsize=(24,12))
ax[0][0].grid(True)
ax[0][0].plot(dimensions, svigp_acc_mean, color=col[0],label=SMCs[0])
ax[0][0].fill_between(dimensions, svigp_acc_mean-Z*svigp_acc_std,svigp_acc_mean+Z*svigp_acc_std, color=col[0],alpha=alp)
ax[0][0].plot(dimensions, svibnn_acc_mean, color=col[1],label=SMCs[1])
ax[0][0].fill_between(dimensions, svibnn_acc_mean-Z*svibnn_acc_std,svibnn_acc_mean+Z*svibnn_acc_std, color=col[1],alpha=alp)
ax[0][0].plot(ep_dimensions, epgp_acc_mean, color=col[2],label=SMCs[2])
ax[0][0].fill_between(ep_dimensions, epgp_acc_mean-Z*epgp_acc_std,epgp_acc_mean+Z*epgp_acc_std, color=col[2],alpha=alp)
ax[0][0].set_title("accuracy (%)")
ax[0][0].legend(loc = 'upper right',prop={'size': 30})
ax[0][0].set_xticks(dimensions)


svigp_unc_mean = np.mean(np.mean(SVI_GP["UNC"],axis=1),axis=1)
svigp_unc_std = np.std(np.std(SVI_GP["UNC"],axis=1),axis=1)
svibnn_unc_mean = np.mean(np.mean(SVI_BNN["UNC"],axis=1),axis=1)
svibnn_unc_std = np.std(np.std(SVI_BNN["UNC"],axis=1),axis=1)
epgp_unc_mean = np.mean(np.mean(EP_GP["UNC"],axis=1),axis=1)
epgp_unc_std = np.std(np.std(EP_GP["UNC"],axis=1),axis=1)

ax[0][1].grid(True)
ax[0][1].plot(dimensions, test_uncertainties, '-.', color=col[3],label='Test')
ax[0][1].plot(dimensions, svigp_unc_mean, color=col[0],label=SMCs[0])
ax[0][1].fill_between(dimensions, svigp_unc_mean-Z*svigp_unc_std,svigp_unc_mean+Z*svigp_unc_std, color=col[0],alpha=alp)
ax[0][1].plot(dimensions, svibnn_unc_mean, color=col[1],label=SMCs[1])
ax[0][1].fill_between(dimensions, svibnn_unc_mean-Z*svibnn_unc_mean,svibnn_unc_mean+Z*svibnn_unc_mean, color=col[1],alpha=alp)
ax[0][1].plot(ep_dimensions, epgp_unc_mean, color=col[2],label=SMCs[2])
ax[0][1].fill_between(ep_dimensions, epgp_unc_mean-Z*epgp_unc_std,epgp_unc_mean+Z*epgp_unc_std, color=col[2],alpha=alp)
ax[0][1].set_title("uncertainty area")
ax[0][1].legend(loc = 'upper right',prop={'size': 30})
ax[0][1].set_xticks(dimensions)


svigp_mse_mean = np.mean(np.mean(SVI_GP["MSE"],axis=1),axis=1)
svigp_mse_std = np.std(np.std(SVI_GP["MSE"],axis=1),axis=1)
svibnn_mse_mean = np.mean(np.mean(SVI_BNN["MSE"],axis=1),axis=1)
svibnn_mse_std = np.std(np.std(SVI_BNN["MSE"],axis=1),axis=1)
epgp_mse_mean = np.mean(np.mean(EP_GP["MSE"],axis=1),axis=1)
epgp_mse_std = np.std(np.std(EP_GP["MSE"],axis=1),axis=1)

ax[1][0].grid(True)
ax[1][0].plot(dimensions, svigp_mse_mean, color=col[0],label=SMCs[0])
ax[1][0].fill_between(dimensions, svigp_mse_mean-Z*svigp_mse_std,svigp_mse_mean+Z*svigp_mse_std, color=col[0],alpha=alp)
ax[1][0].plot(dimensions, svibnn_mse_mean, color=col[1],label=SMCs[1])
ax[1][0].fill_between(dimensions, svibnn_mse_mean-Z*svibnn_mse_std,svibnn_mse_mean+Z*svibnn_mse_std, color=col[1],alpha=alp)
ax[1][0].plot(ep_dimensions, epgp_mse_mean, color=col[2],label=SMCs[2])
ax[1][0].fill_between(ep_dimensions, epgp_mse_mean-Z*epgp_mse_std,epgp_mse_mean+Z*epgp_mse_mean, color=col[2],alpha=alp)
ax[1][0].set_title("root mean square error")
ax[1][0].set_xlabel("dimension")
ax[1][0].legend(loc = 'upper right',prop={'size': 30})
ax[1][0].set_xticks(dimensions)


svigp_time_mean = np.mean(np.mean(SVI_GP["TIME"],axis=1),axis=1)
svigp_time_std = np.std(np.std(SVI_GP["TIME"],axis=1),axis=1)
svibnn_time_mean = np.mean(np.mean(SVI_BNN["TIME"],axis=1),axis=1)
svibnn_time_std = np.std(np.std(SVI_BNN["TIME"],axis=1),axis=1)
epgp_time_mean = np.mean(np.mean(EP_GP["TIME"],axis=1),axis=1)
epgp_time_std = np.std(np.std(EP_GP["TIME"],axis=1),axis=1)

ax[1][1].grid(True)
ax[1][1].plot(dimensions, svigp_time_mean, color=col[0],label=SMCs[0])
ax[1][1].fill_between(dimensions, svigp_time_mean-Z*svigp_time_std,svigp_time_mean+Z*svigp_time_std, color=col[0],alpha=alp)
ax[1][1].plot(dimensions, svibnn_time_mean, color=col[1],label=SMCs[1])
ax[1][1].fill_between(dimensions, svibnn_time_mean-Z*svibnn_time_std,svibnn_time_mean+Z*svibnn_time_std, color=col[1],alpha=alp)
ax[1][1].plot(ep_dimensions, epgp_time_mean, color=col[2],label=SMCs[2])
ax[1][1].fill_between(ep_dimensions, epgp_time_mean-Z*epgp_time_std,epgp_time_mean+Z*epgp_time_std, color=col[2],alpha=alp)
ax[1][1].set_title("training times (hours)")
ax[1][1].legend(loc = 'upper right',prop={'size': 30})
ax[1][1].set_xlabel("dimension")
ax[1][1].set_xticks(dimensions)
plt.tight_layout()
plt.savefig('out/plots/random_performances.png')
plt.close()