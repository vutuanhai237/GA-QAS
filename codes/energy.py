from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
import numpy as np
# from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import UCC,UCCSD, HartreeFock
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit.algorithms.optimizers import SLSQP, NELDER_MEAD, SPSA, L_BFGS_B, P_BFGS, GradientDescent, ADAM, SPSA
from qiskit.algorithms.eigensolvers import VQD
from qiskit.algorithms.state_fidelities import ComputeUncompute, BaseStateFidelity
from qiskit.primitives import Estimator, Sampler, BaseEstimator, BackendEstimator

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit import Aer
#from qiskit.providers.aer import AerSimulator

from qiskit.utils import QuantumInstance
from qiskit import qpy

from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD
import numpy as np
from qiskit_algorithms.utils import algorithm_globals
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit import qpy


def main():
  R = 1.8
  # theta = 90
  basis_set = 'sto6g'

  distance = np.linspace(0.25,2.5,10)
  # distance = np.linspace(70, 110, 10)
  VQE_with_bestcircuit_GAQAS = []
  maxiter = 10000
  seed = 174
# optimizer = NELDER_MEAD(maxiter=100)
  # optimizer = COBYLA(maxiter=500)
  #optimizer = SLSQP(maxiter=maxiter)
  # optimizer = GradientDescent(maxiter=100, learning_rate=0.01)
  # optimizer = SPSA(maxiter=5000, learning_rate=0.001, perturbation=0.001)
  optimizer = ADAM(maxiter=maxiter, tol=1e-06, lr=0.1, beta_1=0.9, beta_2=0.99)
  # optimizer = SLSQP(maxiter=40)
  # optimizer = COBYLA(maxiter=500)


  #file = open(f"H4_chain_30depth_result_energy_{optimizer.__class__.__name__}_{maxiter}_{seed}seed.txt", 'w')
  file = open(f"H4_chain_30depth_result_energy_{optimizer.__class__.__name__}_1_{maxiter}.txt", 'w')

  for i, dis in enumerate(distance):
    theta = dis
    driver = PySCFDriver(
        # atom= f"H 0 {R*np.sin(theta/180*np.pi/2)} {R*np.cos(theta/180*np.pi/2)}; H 0 {-R*np.sin(theta/180*np.pi/2)} {-R*np.cos(theta/180*np.pi/2)}; H 0 {-R*np.sin(theta/180*np.pi/2)} {R*np.cos(theta/180*np.pi/2)}; H 0 {R*np.sin(theta/180*np.pi/2)} {-R*np.cos(theta/180*np.pi/2)} ",
        atom = f"H 0 0 0; H 0 0 {dis}; H 0 0 {2*dis}; H 0 0 {3*dis}",
        # atom = f"H 0 0 0; H 0 0 {dis}",
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
    # mapper = ParityMapper([1,1])

    qubit_op = mapper.map(hamiltonian)

    # for pauli, coeff in sorted(qubit_op.label_iter()):
    #     print(f"{coeff.real:+.8f} * {pauli}")

    #ansatz = UCC(
    #    num_spatial_orbitals= problem.num_spatial_orbitals,
    #    num_particles= [problem.num_alpha, problem.num_beta],
    #    excitations='sd', # single double
    #    reps = 1,
    #    qubit_mapper=mapper,
    #    initial_state = HartreeFock(
    #        num_spatial_orbitals= problem.num_spatial_orbitals,
    #        num_particles= [problem.num_alpha, problem.num_beta],
    #        qubit_mapper=mapper,
    #    ),
    #)


    with open("./8qubits_10points_32circuits_30depth_20generations_VQE_H4_chain_sto6g_fitness_2024-2-12_1000/best_circuit.qpy", "rb") as qpy_file_read:
        ansatz = qpy.load(qpy_file_read)[0]

    # counts = []
    # values = []
    # steps = []

    # def callback(eval_count, params, value, meta):
    #     counts.append(eval_count)
    #     values.append(value)
    #     # steps.append(step)

    algorithm_globals.random_seed = seed

    backend = Aer.get_backend('aer_simulator')
    # quantum_instance = QuantumInstance(backend=backend, shots=None)

    estimator=Estimator()



    vqe = VQE(estimator = estimator, ansatz = ansatz, optimizer=optimizer) #, callback=callback)
    result = vqe.compute_minimum_eigenvalue(qubit_op).eigenvalue.real + nuc

    VQE_with_bestcircuit_GAQAS.append(result)
  #  print(VQE_with_bestcircuit_GAQAS)
    file.write(f"{dis} {result} \n")

  file.close()

    
 


if __name__ == '__main__':
  main()

