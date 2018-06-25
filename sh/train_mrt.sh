#!/usr/bin/env bash

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin:$PATH
export PYTHONPATH=/home/rjq/project/THUMT_mrt:$PYTHONPATH

DATA=~/project/nist_thulac
TRAIN=$DATA/train/
VALID=$DATA/dev_test/nist06/


python /home/rjq/project/THUMT_ruan/thumt/bin/trainer.py \
       --input ${TRAIN}train.zh.shuf ${TRAIN}train.en.shuf \
       --vocabulary ${TRAIN}vocab.zh.32k.txt ${TRAIN}vocab.en.32k.txt \
       --model RNNsearch \
       --validation ${VALID}nist06.cn --references ${VALID}nist06.en0 ${VALID}nist06.en1 ${VALID}nist06.en2 ${VALID}nist06.en3 \
       --parameters device_list=[2],train_steps=300000,MRT=True,batch_size=1,label_smoothing=0.0,dropout=0.0,use_variational_dropout=False,learning_rate=5e-6,learning_rate_decay=none