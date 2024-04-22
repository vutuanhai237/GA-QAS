#!/bin/bash
#PBS -N Test_space
#PBS -o /home/nvlinh/out/
#PBS -e /home/nvlinh/err/
#PBS -l nodes=1:ppn=4
#PBS -q quad  #defines the destination queue of the job.
module load python3.10
cd /home/nvlinh/GA-QAS/codes/
python3.10 Test_space.py
