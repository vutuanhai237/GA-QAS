import sys
sys.path.insert(0, '..')
import numpy as np
import random
import qiskit
import qtm.evolution
import qtm.state
import qtm.qcompilation
import qtm.ansatz
import qtm.constant
from qtm.evolution import environment, mutate, selection, crossover, utils
import matplotlib.pyplot as plt

num_qubits = 6
num_gibbs_qubits = 3
beta = 1
def compilation_gibbs_fitness(qc: qiskit.QuantumCircuit, num_steps = 10):
    compiler1 = qtm.qcompilation.QuantumCompilation(
        # u=qtm.ansatz.g2gnw(4, 2),
        u = qc,
        vdagger=qtm.state.construct_tfd_state(num_gibbs_qubits, beta = 0).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler1.fit(num_steps=num_steps, metrics = ['gibbs'], verbose=0)
    
    compiler2 = qtm.qcompilation.QuantumCompilation(
        # u=qtm.ansatz.g2gnw(4, 2),
        u = qc,
        vdagger=qtm.state.construct_tfd_state(num_gibbs_qubits, beta = 1).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler2.fit(num_steps=num_steps, metrics = ['gibbs'], verbose=0)
    
    compiler3 = qtm.qcompilation.QuantumCompilation(
        # u=qtm.ansatz.g2gnw(4, 2),
        u = qc,
        vdagger=qtm.state.construct_tfd_state(num_gibbs_qubits, beta = 5).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler3.fit(num_steps=num_steps, metrics = ['gibbs'], verbose=0)
    
    return 1./3*(np.average(compiler1.loss_values)+np.average(compiler2.loss_values)+np.average(compiler3.loss_values))

def compilation_gibbs_threshold(fitness_value):
    if fitness_value < 0.1:
        return True
    return False

params = {'depth': 15,
          'num_circuit': 16,  # Must mod 8 = 0
          'num_generation': 10,
          'num_qubits': num_qubits,
          'threshold': compilation_gibbs_threshold,
          'prob_mutate': 0.1}

env2 = environment.EEnvironment(
    params,
    fitness_func = compilation_gibbs_fitness,
    selection_func = selection.elitist_selection,
    crossover_func= crossover.onepoint_crossover,
    mutate_func=mutate.bitflip_mutate,
    pool = qtm.constant.operations,
    file_name="gibbs3"
)
# env2 = environment.EEnvironment('./gibbs4ga_6qubits_compilation_gibbs_fitness_2023-10-28.envobj')
# print(env2.best_candidate.qc)
env2.evol()

