import sys
import matplotlib.pyplot as plt
import qiskit


from qsee.backend import constant
from qsee.evolution import environment, crossover, mutate, selection
import numpy as np, qiskit
from qsee.compilation.qcompilation import QuantumCompilation
from qsee.core import ansatz, state
def compilation_fitness_w(qc: qiskit.QuantumCircuit, num_steps=10):
    """3 qubits => depth 8

    Args:
        qc (qiskit.QuantumCircuit): _description_
        num_steps (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    compiler = QuantumCompilation(
        u=qc,
        vdagger=state.w(4).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, verbose=0)
    return np.min(compiler.loss_values)

def compilation_threshold(fitness_value):
    if fitness_value < 0.1:
        return True
    return False
if __name__ == '__main__':
    import time
    start = time.time()
    params = {'depth': 5,
            'num_circuit': 8,  # Must mod 8 = 0
            'num_generation': 10,
            'num_qubits': 4,
            'threshold': compilation_threshold,
            'prob_mutate': 0.1}

    env = environment.EEnvironment(
        params,
        fitness_func=compilation_fitness_w,
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.bitflip_mutate,
        pool=constant.operations,
        file_name='../experiments/evolution/'
    )


    env.evol()
    print(time.time() - start)