from qsee.backend import utilities
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import state
import qiskit
import time 

num_qubits = 2 
# st = time.time() 

qc = utilities.load_circuit('/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_100_4qubits_compilation_fitness_gibbs_2024-01-18/best_circuit')

# qsp = QuantumStatePreparation(
#     u=qc,
#     target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
# ).fit(100)
# dur = time.time() - st   
# print("time compilation:",dur) 
# time compilation: 310.51073384284973
import qiskit_algorithms
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
# from qiskit_nature.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.units import DistanceUnit
from qiskit.utils import algorithm_globals

st = time.time() 
optimizer = qiskit_algorithms.optimizers.COBYLA(maxiter=100)
estimator = Estimator()
#algorithm_globals.random_seed = 50

driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.6",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
problem = driver.run()
hamiltonian = problem.hamiltonian.second_q_op()
algorithm_globals.random_seed = 50

mapper=JordanWignerMapper()
qubit_op = mapper.map(hamiltonian)
vqe = VQE(estimator = estimator, ansatz = qc, optimizer=optimizer)
ene_vqe = vqe.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
print(ene_vqe)

dur = time.time() - st   
print("time VQE:",dur) 
#time VQE: 3.14336895942688