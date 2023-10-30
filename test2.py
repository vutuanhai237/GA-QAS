import concurrent.futures
import time, typing
import qiskit     
import numpy as np
from testssss import A
from qtm.evolution.ecircuit import ECircuit
import qtm.random_circuit
def bypass_compile(circuit: ECircuit):
    circuit.compile()
    print(circuit.fitness)
    return circuit
def multiple_compile(circuits: typing.List[ECircuit]):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(bypass_compile, circuits)
    return results

def compilation_fitness_ghz(qc: qiskit.QuantumCircuit, num_steps=5):
    compiler = qtm.qcompilation.QuantumCompilation(
        u=qc,
        vdagger=qtm.state.create_ghz_state(3).inverse(),
        optimizer='adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=num_steps, verbose=0)
    return np.average(compiler.loss_values)

if __name__ == '__main__':
    start = time.time()
    random_circuit = qtm.random_circuit.generate_with_pool(
                3, 3)
    a1 = ECircuit(random_circuit, compilation_fitness_ghz)
    a2 = ECircuit(random_circuit, compilation_fitness_ghz)
    a3 = ECircuit(random_circuit, compilation_fitness_ghz)
    numbers = [a1, a2, a3]
    
    results = multiple_compile(numbers)
    for a in numbers:
        print(a.fitness)
    end = time.time()
    print(end - start)
