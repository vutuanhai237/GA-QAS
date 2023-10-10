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
qc_haar = qtm.state.create_haar_state(3)


def compilation_fitness(qc: qiskit.QuantumCircuit, num_steps=5):
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qc,
        vdagger=qc_haar.inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, verbose=0)
    return np.average(compiler.loss_values)


def compilation_threshold(fitness_value):
    if fitness_value < 0.4:
        return True
    return False


params = {'depth': 5,
          'num_circuit': 8,  # Must mod 8 = 0
          'num_generation': 2,
          'num_qubits': 3,
          'threshold': compilation_threshold,
          'prob_mutate': 0.01}

env = environment.EEnvironment(
    params,
    fitness_func = compilation_fitness,
    selection_func = selection.elitist_selection,
    crossover_func= crossover.onepoint_crossover,
    mutate_func=mutate.bitflip_mutate,
    pool = qtm.constant.operations
)


env.evol()
env.save('./experiments/evolution/test.envobj')