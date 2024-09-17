#!/bin/bash
#PBS -l nodes=1:ppn=32 -q gpu
python HT/HTProject/ht_nsf_project.py htrisk train htrisk_models/roberta_no_special_tokens_run3 data/htrp_no_special_tokens
