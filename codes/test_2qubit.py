import sys, qiskit
sys.path.insert(0, '..')
import numpy as np, os, pandas as pd
import qiskit.quantum_info as qi
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import state, measure
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
from funcs import create_params

m = 5
n = 4
def testing(n, d, n_circuit, n_gen):
    utests = []
    print("Testing states:")
    for i in range(0, m):
        utest = state.haar(n)
        print(qi.Statevector.from_instruction(utest).data)
        utests.append(utest)
    def changeRisk(n, d, n_circuit, n_gen, risk):
        df = pd.read_csv('risk.csv')
        filtered_df = df[(df['n'] == n) & (df['d'] == d) & (df['n_circuit'] == n_circuit) & (df['n_gen'] == n_gen)]
        row_index = filtered_df.index.tolist()[0]
        df.loc[row_index] = [n, d, n_circuit, n_gen, risk, df.loc[row_index]['cost']]
        df.to_csv(f'risk{n}.csv', index=False)
        return
    def random_compiltion_test(qc_best: qiskit.QuantumCircuit):
        risks = []
        for i in range(0, m):
            qsp = QuantumStatePreparation(
                u=qc_best,
                target_state=utests[i].inverse()
            ).fit(num_steps=100)
            risks.append((measure.measure(utests[i]) - measure.measure(qc_best.assign_parameters(qsp.compiler.thetas)))**2)
        return np.mean(risks)/4
    env = EEnvironment.load(f'n={n},d={d},n_circuit={n_circuit},n_gen={n_gen}', None)
    # env = EEnvironment.load(f'n={2},d={2},n_circuit={4},n_gen={10}', None)
    risk = (random_compiltion_test(env.best_circuit))
    changeRisk(n, d, n_circuit, n_gen, risk)

def multiple_compile(params):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile, params)
    return results

def bypass_compile(param):
    d, n_circuit, n_gen = param
    if os.path.isdir(f'n={n},d={d},n_circuit={n_circuit},n_gen={n_gen}'):
        print(n, d, n_circuit, n_gen)
        testing(n, d, n_circuit, n_gen)
if __name__ == '__main__':
    depths = list(range(2, 5)) # 2 qubits case
    num_circuits = [4, 8, 16, 32]
    num_generations = [10, 20, 30, 40, 50]
    params = create_params(depths, num_circuits, num_generations)
    multiple_compile(params)