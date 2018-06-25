#!/usr/bin/env bash

DATA=~/project/nist_thulac
VALID=$DATA/dev_test/nist06/
TEST=/home/rjq/project/nist_thulac/dev_test/nist02/

perl multi-bleu.perl ${TEST}nist02.en0 ${TEST}nist02.en1 ${TEST}nist02.en2 ${TEST}nist02.en3 < infer/nist02.en.trans