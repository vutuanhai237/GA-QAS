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

    
def superevol(num_qubits,depth):
    
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
        
        return 1 - np.min(qsp1.compiler.metrics['compilation_trace_fidelities'])+np.min(
            qsp2.compiler.metrics['compilation_trace_fidelities'])+np.min(qsp3.compiler.metrics['compilation_trace_fidelities'])
    
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
    env.evol()
    
    qc = env.best_circuit
    
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
    
    with open(f'./experiments/gibbs{num_qubits}qubits.pickle', 'wb') as handle:
        pickle.dump([qsp1,qsp2,qsp3], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(f'./experiments/gibbs{num_qubits}qubits.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
    #     print(b[0],b[1],b[2])

import concurrent.futures
import math

# superevol(2,9)
it_parameters = list(range(2,7))
print(it_parameters)

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number in (executor.map(superevol, it_parameters,[9,15,21,27,33])):
            print((number))

if __name__ == '__main__':
    main()