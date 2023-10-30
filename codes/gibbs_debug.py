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

num_qubits = 2
num_gibbs_qubits = 3
beta = 1
def compilation_gibbs_fitness(qc: qiskit.QuantumCircuit, num_steps = 10):
    compiler = qtm.qcompilation.QuantumCompilation(
        # u=qtm.ansatz.g2gnw(4, 2),
        u = qc,
        vdagger=qtm.state.create_ghz_state(2).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, metrics = ['gibbs'], verbose=0)
    return np.average(compiler.loss_values)
def compilation_gibbs_threshold(fitness_value):
    if fitness_value < 0.1:
        return True
    return False

params = {'depth': 4,
          'num_circuit': 3,  # Must mod 8 = 0
          'num_generation': 10,
          'num_qubits': num_qubits,
          'threshold': compilation_gibbs_threshold,
          'prob_mutate': 0.1}

env = environment.EEnvironment(
    params,
    fitness_func = compilation_gibbs_fitness,
    selection_func = selection.elitist_selection,
    crossover_func= crossover.onepoint_crossover,
    mutate_func=mutate.bitflip_mutate,
    pool = qtm.constant.operations
)

env.evol()
