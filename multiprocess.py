import numpy as np
import qiskit
import qiskit.quantum_info as qi
from qoop.core import state, metric, ansatz
from qoop.compilation.qcompilation import QuantumCompilation
from qoop.backend import utilities


n_test = 10

def multiple_compile(params):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile, params)
    return results

def graph(qc, random_state):
    compiler =  QuantumCompilation(
        u = qc,
        vdagger = state.specific(random_state).inverse(),
        metrics_func=[
            'compilation_trace_fidelities'
        ]
        
    )
    
    compiler.fast_fit(100) 
    return compiler.metrics['compilation_trace_fidelities'][-1]

def bypass_compile(num_qubits):
    fidelitiess = []
    tests = []
    if num_qubits == 2:
        depth = [3]
    elif num_qubits == 3:
        depth = range(5,15)
    elif num_qubits == 4:
        depth = range(5, 30)
    elif num_qubits == 5:
        depth = range(5, 40)
    elif num_qubits == 6:
        depth = range(5, 40)
    index = 0
    for i in range(0, n_test):
        random_state = np.random.uniform(0, 2*np.pi, 2**num_qubits)
        random_state = random_state/np.linalg.norm(random_state)
        tests.append(np.array(random_state))
    while(index < len(depth)):
        fidelities = []
        qc = utilities.load_circuit(f'data/n={num_qubits},d={depth[index]},n_circuit=32,n_gen=50/best_circuit')
        for i in range(0, n_test):
            fidelities.append(graph(qc, tests[i]))
        fidelitiess.append(np.mean(fidelities))
        
    print(f"---n={num_qubits}---")
    print(fidelitiess)
    return fidelitiess
    # if np.mean(fidelities) < 0.99:
    #     index += 1
    #     if index == len(depth):
    #         break
    # else:
    #     break

if __name__ == '__main__':
    ns = [2,3,4]
    multiple_compile(ns)