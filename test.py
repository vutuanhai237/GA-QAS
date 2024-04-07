import sys, qiskit
import matplotlib.pyplot as plt
import numpy as np
import csv
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit, metric
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
import pickle
import scipy
from numpy import trace as np_trace
from qiskit.quantum_info import Pauli, Operator
import qiskit.quantum_info as qi
hamiltonian= qiskit.quantum_info.SparsePauliOp(['ZZ', 'IZ', 'ZI'], [-1, -1, -1]).to_matrix() 

def gibbs_state(beta): 
    # Generate the ideal target Gibbs state rho
    rho_G = scipy.linalg.expm(-1 * beta * hamiltonian) / np_trace(scipy.linalg.expm(-1 * beta * hamiltonian))
    
    return rho_G

# # Now you reconstruct the compiler
betas = np.linspace(0,1,5)
#load best circuit
loaded_qc = ansatz.g2gn(2,1)#utilities.load_circuit('n=2,d=4,n_circuit=4,n_gen=8/best_circuit')

qsp_list = []
#get theta values
for beta in betas:
    qc_time = np.asarray(gibbs_state(beta))
    print(qc_time)
    qsp = QuantumStatePreparation(
        u=loaded_qc,
        target_state=qc_time
    ).fit(num_steps=100)
    print(qi.Statevector.from_instruction(qsp.vdagger))
    print(qsp.compiler.metrics['loss_fubini_study'][-1])
    qsp_list.append(qsp)

for i in range(len(qsp_list)):
    print(qsp_list[i].compiler.metrics['loss_fubini_study'][-1])
    plt.plot(i,qsp_list[i].compiler.metrics['loss_fubini_study'][-1],label='theory')

# Define the file name
csv_file = 'gibbs_thetass.csv'

# Extract data from each QuantumStatePreparation object
data = []
for qsp in qsp_list:
    # Extract relevant data, such as theta values
    theta_values = qsp.thetas 
    data.append(theta_values)

# Write the data to a CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)