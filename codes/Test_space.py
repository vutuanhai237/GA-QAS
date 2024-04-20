from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
import numpy as np
# from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCC,UCCSD, HartreeFock
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_algorithms.optimizers import SLSQP, NELDER_MEAD, SPSA, L_BFGS_B, P_BFGS, GradientDescent, ADAM, SPSA
from qiskit_algorithms.eigensolvers import VQD
from qiskit_algorithms.state_fidelities import ComputeUncompute, BaseStateFidelity
from qiskit.primitives import Estimator, Sampler, BaseEstimator, BackendEstimator
# from qiskit_aer.primitives import Estimator
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit import Aer
#from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit_algorithms.utils import algorithm_globals
from qiskit.algorithms.minimum_eigensolvers import VQE
#from qiskit_algorithms import VQE
from qiskit import qpy
import json
import concurrent.futures
#executor = concurrent.futures.ProcessPoolExecutor()


# distance = np.linspace(0.5,2.0,40)
#distance = np.linspace(0.25, 2.5, 10)
#distance = [1.5]
# distance = [2.5]
# distance = np.linspace(70, 110, 10)





def VQE_Test(dis):
  data = {'seed': [], 'energy': [], 'thetas': []}
  interation = 1000
  seeds = np.arange(1,32+1)
  basis_set = 'sto6g'
  theta = dis
  driver = PySCFDriver(
      # atom= f"H 0 {R*np.sin(theta/180*np.pi/2)} {R*np.cos(theta/180*np.pi/2)}; H 0 {-R*np.sin(theta/180*np.pi/2)} {-R*np.cos(theta/180*np.pi/2)}; H 0 {-R*np.sin(theta/180*np.pi/2)} {R*np.cos(theta/180*np.pi/2)}; H 0 {R*np.sin(theta/180*np.pi/2)} {-R*np.cos(theta/180*np.pi/2)} ",
      atom = f"H 0 0 0; H 0 0 {dis}; H 0 0 {2*dis}; H 0 0 {3*dis}",
      #atom = "H 0 0 0; H 0 0 {dis}",
      basis= basis_set,
      charge=0,
      spin=0,
      unit=DistanceUnit.ANGSTROM,
  )

  problem = driver.run()
  hamiltonian = problem.hamiltonian
  nuc = hamiltonian.nuclear_repulsion_energy
  hamiltonian = hamiltonian.second_q_op()

  mapper = JordanWignerMapper()

  qubit_op = mapper.map(hamiltonian)


  with open(f"./8qubits_10points_16circuits_21depth_20generations_VQE_H4_chain_sto6g_fitness_2024-1-25_1000/best_circuit.qpy", "rb") as qpy_file_read:
      ansatz = qpy.load(qpy_file_read)[0]

  # def callback(eval_count, params, value, meta):
  #     counts.append(eval_count)
  #     values.append(value)
  #     print(value + nuc)


  # optimizer = NELDER_MEAD(maxiter=100)
  # optimizer = COBYLA(maxiter=10000)
  optimizer = SLSQP(maxiter=interation)
  # optimizer = GradientDescent(maxiter=100, learning_rate=0.01)
  # optimizer = SPSA(maxiter=5000, learning_rate=0.001, perturbation=0.001)
  # optimizer = ADAM(maxiter=1000, tol=1e-06, lr=0.02, beta_1=0.9, beta_2=0.99)

  for i, seed in enumerate(seeds):
    algorithm_globals.random_seed = seed 
    estimator = Estimator()
    vqe = VQE(estimator = estimator, ansatz = ansatz, optimizer=optimizer, callback=None)
    result1 = vqe.compute_minimum_eigenvalue(qubit_op)
    result = result1.eigenvalue.real + nuc
    # print(result1.optimal_parameters)
    # print(result1.optimal_point)
    # print(result1.optimal_circuit)
    data['seed'].append(int(seed))
    data['energy'].append(float(result))
    data['thetas'].append(list(result1.optimal_point))
    print(data)
  with open(f"result_H4_21depth_SLSQP_{dis}_{interation}.json", "w") as outfile:
    json.dump(data, outfile)
  
if __name__ == '__main__':
  distance = np.linspace(0.25, 2.5, 10)
  #distance = [0.25]
  #import multiprocessing
  executor = concurrent.futures.ProcessPoolExecutor()
  executor.map(VQE_Test, distance)	
  #with multiprocessing.Pool() as pool:
  #  pool.map(VQE_Test, distance)
  #for i,dis in enumerate(distance):
  #  VQE_Test(dis)
	

