import sys, qiskit
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
from qsee.compilation import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata

def superevol(num_qubits):
    def compilation_fitness_ghz(qc: qiskit.QuantumCircuit):
        qsp = QuantumStatePreparation(
            u=qc,
            target_state=state.ghz(num_qubits).inverse()
        ).fit(num_steps=10)
        return 1 - np.min(qsp.compiler.metrics['loss_fubini_study'])

    def full_compilation_fitness_ghz(qc: qiskit.QuantumCircuit):
        qsp = QuantumStatePreparation(
            u=qc,
            target_state=state.ghz(num_qubits).inverse()
        ).fit(num_steps=100)
        return 1 - np.min(qsp.compiler.metrics['loss_fubini_study'])
    
    depth = 4
    num_circuit = 16
    env_metadata = EEnvironmentMetadata(
        num_qubits = num_qubits,
        depth = depth,
        num_circuit = num_circuit,
        num_generation = 20,
        prob_mutate=3/(depth * num_circuit)
    )
    env = EEnvironment(
        metadata = env_metadata,
        fitness_func=[compilation_fitness_ghz, full_compilation_fitness_ghz],
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.layerflip_mutate,
        threshold_func=threshold.compilation_threshold
    )
    env.evol()

import concurrent.futures
import math

it_parameters = list(range(4, 11))


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number in (executor.map(superevol, it_parameters)):
            print((number))

if __name__ == '__main__':
    main()