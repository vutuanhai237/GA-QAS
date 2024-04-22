import qiskit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit_algorithms.utils import algorithm_globals
from qiskit.algorithms.optimizers import SLSQP, COBYLA

def general_VQE(qc: qiskit.QuantumCircuit, atom: str, basis: str):

    
    driver = PySCFDriver(
        atom=atom,
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    hamiltonian = problem.hamiltonian
    nuc = hamiltonian.nuclear_repulsion_energy
    #optimizer = SLSQP(maxiter=200)
    optimizer = COBYLA(maxiter=1000)
    estimator = Estimator()
    algorithm_globals.random_seed = 50
    mapper=JordanWignerMapper()
    qubit_op = mapper.map(hamiltonian.second_q_op())
    vqe = VQE(estimator = estimator, ansatz = qc, optimizer=optimizer)
    computation_value = vqe.compute_minimum_eigenvalue(qubit_op).eigenvalue.real + nuc
    return computation_value
