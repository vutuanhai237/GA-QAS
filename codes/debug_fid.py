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

num_qubits = 2
beta = 1

compiler = qtm.qcompilation.QuantumCompilation(
    u=qtm.ansatz.g2gnw(4, 2),
    vdagger=qtm.state.construct_tfd_state(num_qubits,beta = 1).inverse(),
    optimizer='adam',
    loss_func='loss_fubini_study'
)
compiler.fit(num_steps=1, verbose=1)

print(compiler.gibbs_fidelities)