import sys
sys.path.insert(0, '..')
import numpy as np
import qiskit
import qtm.evolution
import qtm.state
import qtm.qcompilation
import qtm.ansatz
import qtm.constant
from qtm.evolution import environment, mutate, selection, crossover, utils
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2
num_qubits = 2

if num_qubits == 2:
    qc_haar = qtm.state.create_specific_state(num_qubits,
    [ 0.73042448, -0.30956267, -0.45658101,  0.40272176])

def compilation_fitness_w(qc: qiskit.QuantumCircuit, num_steps=5):
    """3 qubits => depth 8

    Args:
        qc (qiskit.QuantumCircuit): _description_
        num_steps (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qc,
        vdagger=qtm.state.create_w_state(num_qubits).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, verbose=0)
    return np.average(compiler.loss_values)

def compilation_fitness_ghz(qc: qiskit.QuantumCircuit, num_steps=5):
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qc,
        vdagger=qtm.state.create_ghz_state(num_qubits).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, verbose=0)
    return np.average(compiler.loss_values)

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
    if fitness_value < 0.1:
        return True
    return False

params = {'depth': 4,
          'num_circuit': 8,  # Must mod 8 = 0
          'num_generation': 10,
          'num_qubits': num_qubits,
          'threshold': compilation_threshold,
          'prob_mutate': 0.1}
if __name__ == "__main__":

    env = environment.EEnvironment(
        params,
        fitness_func=compilation_fitness_ghz,
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.bitflip_mutate,
        pool=qtm.constant.operations,
        file_name='./experiments/evolution/'
    )

    env.evol()