import numpy as np
import random
import qiskit
import qtm.evolution
import qtm.state
import qtm.qcompilation
import qtm.ansatz
import qtm.constant
from qtm.evolution import environment, mutate, selection, crossover
import matplotlib.pyplot as plt
qc_haar = qtm.state.create_haar_state_inverse(3)
def compilation_fitness(qc: qiskit.QuantumCircuit):
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qc,
        vdagger=qc_haar,
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=10, verbose=0)
    return np.average(compiler.loss_values)

params = {'depth': 4,
          'num_individual': 8,  # Must mod 8 = 0
          'num_generation': 5,
          'num_qubits': 3,
          'threshold': 0.2,
          'prob_mutate': 0.01}

env = environment.EEnvironment(
    params,
    fitness_func = compilation_fitness,
    selection_func = selection.elitist_selection,
    crossover_func= crossover.onepoint_crossover,
    mutate_func=mutate.bitflip_mutate,
    pool = qtm.constant.operations
)

env.initialize_population()
env.evol()