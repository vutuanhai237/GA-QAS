import concurrent.futures
import time
import qiskit     
import numpy as np
import sys
sys.path.insert(0, '..')
import qtm.qcompilation, qtm.ansatz, qtm.state


class A:
    def __init__(self):
        self.t = 0
    def run(self):
        self.t = test_nqubit_tomography()
def test_nqubit_tomography():
    num_qubits = 3
    num_layers = 1
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qtm.state.create_ghz_state(num_qubits).inverse(),
        vdagger = qtm.ansatz.Wchain_ZXZlayer_ansatz(num_qubits, num_layers),
        optimizer = 'adam',
        loss_func='loss_fubini_study'
    )
    compiler.fit(num_steps=10, verbose = 1)
    return compiler.loss_values[-1]

def func1(a: A):
    a.run()
    return a
def main(numbers):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(func1, numbers)
        return results
        

if __name__ == '__main__':
    start = time.time()
    a1 = A()
    a2 = A()
    a3 = A()
    numbers = [a1, a2, a3]
    
    results = main(numbers)
    for a in results:
        print(a.t)
    end = time.time()
    print(end - start)
