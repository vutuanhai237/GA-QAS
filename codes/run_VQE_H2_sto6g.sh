#!/bin/bash
cd /home/nvlinh/GA-QAS/codes/
#PBS -N GA-QAS-molecules # đặt tên cho job
#PBS -o /home/nvlinh/out
#PBS -e /home/nvlinh/err
#PBS -l nodes=1:ppn=8
#PBS -q octa
module load python3.10
python3.10 qevocircuit_VQE_H2_sto6g.py
