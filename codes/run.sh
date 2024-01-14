#!/bin/bash
#PBS -N GA-QAS-molecules
#PBS -l walltime=12:00:00
#PBS -o /home/nvlinh/out/
#PBS -e /home/nvlinh/err/
#PBS -l nodes=1:ppn=8
#PBS -q octa #defines the destination queue of the job.
module load python3.10
cd /home/nvlinh/GA-QAS/codes/
python3.10 qevocircuit_VQE_H2_sto6g.py
