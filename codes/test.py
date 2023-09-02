import numpy as np
import random
import types
import qiskit
import qtm.evolution
import random_circuit
import qtm.state
import qtm.qcompilation
import qtm.ansatz

print(random_circuit.random_circuit2(3, 5).depth())