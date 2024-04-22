#!/bin/bash
#PBS -N GA-QAS-molecules
#PBS -o /home/nvlinh/out/
#PBS -e /home/nvlinh/err/
#PBS -l nodes=1:ppn=8
#PBS -q octa #defines the destination queue of the job.
module load python3.10
cd /home/nvlinh/GA-QAS/codes/
python3.10 qevocircuit_VQE_H4_chain_sto6g.py
