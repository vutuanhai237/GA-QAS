import sys
import matplotlib.pyplot as plt
import qiskit
from qsee.backend import constant
from qsee.evolution import environment, crossover, mutate, selection
import numpy as np, qiskit
from qsee.compilation import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit

qc = random_circuit.generate_with_pool(3, 5)
qsp = QuantumStatePreparation(
    u=qc,
    target_state=state.w(3).inverse()
).fit(3)