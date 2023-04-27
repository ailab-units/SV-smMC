#!/bin/bash

############
# settings #
############

CATHEGORY="random"
N_POSTERIOR_SAMPLES=1000

SVI_GP_VARIATIONAL_DISTRIBUTION="cholesky"
SVI_GP_VARIATIONAL_STRATEGY="default"
SVI_GP_BATCH_SIZE=100 
SVI_GP_N_EPOCHS=2000
SVI_GP_LR=0.001

SVI_BNN_ARCHITECTURE="3L"
SVI_BNN_BATCH_SIZE=100
SVI_BNN_N_EPOCHS=2000
SVI_BNN_LR=0.001
SVI_BNN_N_HIDDEN=30

#######
# run #
#######

source venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"

python EP_GPs/train.py --cathegory=$CATHEGORY >> $OUT 2>&1

python SVI_BNNs/train.py --cathegory=$CATHEGORY --architecture=$SVI_BNN_ARCHITECTURE --batch_size=$SVI_BNN_BATCH_SIZE \
	--n_epochs=$SVI_BNN_N_EPOCHS --lr=$SVI_BNN_LR --n_hidden=$SVI_BNN_N_HIDDEN \
	--device="cpu">> $OUT 2>&1

python SVI_GPs/train.py --cathegory=$CATHEGORY --variational_distribution=$SVI_GP_VARIATIONAL_DISTRIBUTION \
	--variational_strategy=$SVI_GP_VARIATIONAL_STRATEGY --batch_size=$SVI_GP_BATCH_SIZE \
	--n_epochs=$SVI_GP_N_EPOCHS --lr=$SVI_GP_LR --device="cuda" >> $OUT 2>&1