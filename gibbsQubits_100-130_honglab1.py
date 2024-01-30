import sys, qiskit
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
import qsee
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
import pickle

#11
#223
#930
#4023
#16277
def superevol(num_qubits,depth,i=100):

    def compilation_fitness_more_than_2_gibbs(qc: qiskit.QuantumCircuit):
        qsp1 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
        ).fit(num_steps=10)
        
        #(max_fid1+max_fid2+max_fid3)
        return 1 - qsp1.compiler.metrics['loss_fubini_study'][-1]

    def compilation_fitness_gibbs(qc: qiskit.QuantumCircuit):
        qsp1 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
        ).fit(num_steps=10)
        
        qsp2 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 1).inverse()
        ).fit(num_steps=10)
                
        qsp3 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 5).inverse()
        ).fit(num_steps=10)
        
        qsp4 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 7).inverse()
        ).fit(num_steps=10)
        
        qsp5 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 10).inverse()
        ).fit(num_steps=10)
                
        #(max_fid1+max_fid2+max_fid3)
        return 1-np.average([qsp1.compiler.metrics['loss_fubini_study'][-1],
            qsp2.compiler.metrics['loss_fubini_study'][-1],qsp3.compiler.metrics['loss_fubini_study'][-1],
            qsp4.compiler.metrics['loss_fubini_study'][-1],
            qsp5.compiler.metrics['loss_fubini_study'][-1]]) 

    def full_compilation_fitness_gibbs(qc: qiskit.QuantumCircuit):
        #ve hinh density matrix cho n=2, toi beta=8,10, th khac: so sanh kq fid, purity:  0,1,2,3,4,5,6,7,8,9,10
        #co dinh beta=0, thay doi n, eps
         
        qsp1 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
        ).fit(num_steps=100)        
        
        qsp2 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 1).inverse()
        ).fit(num_steps=100)
        
        qsp3 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 5).inverse()
        ).fit(num_steps=100)
        
        qsp4 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 7).inverse()
        ).fit(num_steps=100)
        
        qsp5 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 10).inverse()
        ).fit(num_steps=100)
                
        return 1-np.average([qsp1.compiler.metrics['loss_fubini_study'][-1],
            qsp2.compiler.metrics['loss_fubini_study'][-1],qsp3.compiler.metrics['loss_fubini_study'][-1],
            qsp4.compiler.metrics['loss_fubini_study'][-1],
            qsp5.compiler.metrics['loss_fubini_study'][-1]]) 
            
    num_circuit = 16
    env_metadata = EEnvironmentMetadata(
        num_qubits = 2*num_qubits,
        depth = depth,
        num_circuit = num_circuit,
        num_generation = 20,
        prob_mutate=3/(depth * num_circuit)
    )
    env = EEnvironment(
        metadata = env_metadata,
        fitness_func= compilation_fitness_gibbs,
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.layerflip_mutate,
        threshold_func=threshold.compilation_threshold
    )
    env.set_filename(f'Trial_{i}')
    env.evol()

def multiple_compile(params):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile, params)
    return results

def bypass_compile(i):
    superevol(2,9,i=i)

if __name__ == '__main__':
    iss = list(range(130, 160))
    multiple_compile(iss)
    
for i in range(0,100):
    superevol(2,9,i=i)
