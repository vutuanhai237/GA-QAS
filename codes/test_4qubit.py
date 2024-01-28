import sys, qiskit
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np, os, json
import qiskit.quantum_info as qi
import pandas as pd
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state, measure
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
from funcs import create_params, calculate_risk

n = 4
def changeRisk(n, d, n_circuit, n_gen, risks):
    with open(f'risk_{n}.json', 'r') as file:
        data = json.load(file)

    # Update the data with the new key and list of values
    data[f'n={n},d={d},n_circuit={n_circuit},n_gen={n_gen}'] = risks

    # Open the JSON file in write mode ('w') and write the updated data
    with open(f'risk_{n}.json', 'w') as file:
        json.dump(data, file)
    return
def test(n, d, n_circuit, n_gen):
    with open(f'risk_{n}.json', 'r') as file:
        data = json.load(file)
    case = f'n={n},d={d},n_circuit={n_circuit},n_gen={n_gen}'
    if case in data.keys():
        return
    print(n, d, n_circuit, n_gen)
    n_train = 20
    utrains = []
    for i in range(0, n_train):
        utrain = state.haar(n)
        utrains.append(utrain)
    n_test = 10
    utests = []
    for i in range(0, n_test):
        utest = state.haar(n)
        utests.append(utest)
    env = EEnvironment.load(case, None)
    best_circuit = env.best_circuit
    risks = []
    for i in range(0, n_train):
        qsp = QuantumStatePreparation(
            u=best_circuit,
            target_state=utrains[i].inverse()
        ).fit(num_steps=100)
        risk = calculate_risk(utests, best_circuit.assign_parameters(qsp.thetas))
        risks.append(risk)
    changeRisk(n, d, n_circuit, n_gen, risks)
    return risks

def multiple_compile(params):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor(4)
    results = executor.map(bypass_compile, params)
    return results

def bypass_compile(param):
    d, n_circuit, n_gen = param
    if os.path.isdir(f'n={n},d={d},n_circuit={n_circuit},n_gen={n_gen}'):
        test(n, d, n_circuit, n_gen)
if __name__ == '__main__':
    depths = list(range(5, 40)) # 4 qubits case
    num_circuits = [4, 8, 16, 32]
    num_generations = [10, 20, 30, 40, 50]
    params = create_params(depths, num_circuits, num_generations)
    multiple_compile(params)