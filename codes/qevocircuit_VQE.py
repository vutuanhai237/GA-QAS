import sys, qiskit, typing
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.vqe import vqe, utilities
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
#%load_ext autoreload
#%autoreload 2

h2_631g = lambda distances:  f"H 0 0 {distances[0]}; H 0 0 {distances[1]}"
h4_sto3g = lambda distances: f"H 0 0 {distances[0]}; H 0 0 {distances[1]}; H 0 0 {distances[2]}; H 0 0 {distances[3]}"
lih_sto3g = lambda distances: f"Li 0 0 {distances[0]}; H 0 0 {distances[1]}"

file = open('Hydrogen molecules-sto3g-200.txt', 'r').readlines()

def exact_VQE_H2_631g(distances: []):
    return 

def general_VQE_H2_631g(distances: []):
    def VQE_H2_631g(qc: qiskit.QuantumCircuit):
        return VQE_fitness(qc, 
                           # Replace atom here, below function returns text such as "H 0 0 0; H 0 0 0.25"
                           atom = h2_631g(distances), 
                           # Replace basis here
                           basis = "sto3g") #631g
    return VQE_H2_631g

def VQE_fitness(qc: qiskit.QuantumCircuit, atom: str, basis: str) -> float:
    """General VQE fitness

    Args:
        qc (qiskit.QuantumCircuit): ansatz
        atom (str): describe for atom
        basis (str): VQE basis

    Returns:
        float: similarity between experiment results and theory results
    """
    computation_value = vqe.general_VQE(qc, atom, basis)
    # I need to modify this
    exact_value = eval(file[round((eval(atom.split(' ')[-1])-0.25)/((2.5-0.25)/199))].split(" ")[-2])
    return utilities.similarity(computation_value, exact_value)

def VQE_H2_631g_fitness(qc: qiskit.QuantumCircuit) -> float:
    """Fitness function for H2_631g case

    Args:
        qc (qiskit.QuantumCircuit): ansatz

    Returns:
        float: fitness value
    """
    num_points = 6
    # Create pairs of distanc
    list_distances_H2_631g = list(zip([0]*num_points, np.linspace(0.5, 2.5, num_points))) 
    fitnesss = []
    # Run for num_points
    for distances in list_distances_H2_631g:
        # Below is fitness function at special point, this function has only qc as parameter
        specific_VQE_H2_631g: typing.FunctionType = general_VQE_H2_631g(distances)
        fitnesss.append(specific_VQE_H2_631g(qc))
    return np.mean(fitnesss)



env_metadata = EEnvironmentMetadata(
    num_qubits=4,
    depth=3,
    num_circuit=4,
    num_generation=5,
    prob_mutate=3/(3 * 4)  # Mutation rate / (depth * num_circuit)
)
env = EEnvironment(
    metadata=env_metadata,
    # Fitness function alway has the function type: qiskit.QuantumCircuit -> float
    fitness_func=VQE_H2_631g_fitness,
    selection_func=selection.elitist_selection,
    crossover_func=crossover.onepoint_crossover,
    mutate_func=mutate.layerflip_mutate,
    threshold_func=threshold.compilation_threshold
)

# Automatically save the results in the same level folder
env.evol(1)

computation_value = vqe.general_VQE(env.best_circuit, h2_631g([0,0.25]), basis='sto3g')
print(computation_value)

# Load the result from folder
env2 = EEnvironment.load(
    './4qubits_VQE_H2_sto3g_fitness_2023-12-30', 
    VQE_H2_631g_fitness
)

env2.plot()
