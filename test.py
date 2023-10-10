import sys
import qtm.constant
import qtm.ansatz
import qtm.qcompilation
import qtm.state
import qtm.evolution.environment
import qiskit
import random
import numpy as np
qc_haar = qtm.state.create_haar_state(2)

def compilation_fitness(qc: qiskit.QuantumCircuit, num_steps=5):
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qc,
        vdagger=qc_haar,
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, verbose=0)
    return np.average(compiler.loss_values)


def compilation_threshold(fitness_value):
    if fitness_value < 0.4:
        return True
    return False

env2 = qtm.evolution.environment.EEnvironment('./codes/ga_2qubits_compilation_fitness_2023-10-08.envobj')
env2.evol()