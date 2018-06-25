#!/usr/bin/env bash

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-8.0/bin:$PATH
export PYTHONPATH=/home/rjq/project/THUMT_mrt:$PYTHONPATH

DATA=~/project/nist_thulac

TRAIN=$DATA/train/
VALID=$DATA/dev_test/nist06/


python /home/rjq/project/THUMT_mrt/thumt/bin/trainer.py \
       --input ${TRAIN}train.zh.shuf ${TRAIN}train.en.shuf \
       --vocabulary ${TRAIN}vocab.zh.32k.txt ${TRAIN}vocab.en.32k.txt \
       --model transformer \
       --validation ${VALID}nist06.cn --references ${VALID}nist06.en0 ${VALID}nist06.en1 ${VALID}nist06.en2 ${VALID}nist06.en3 \
       --parameters device_list=[0],batch_size=6000,beam_size=10,train_steps=300000