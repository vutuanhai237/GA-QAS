import qiskit 
from qiskit.quantum_info import SparsePauliOp
import numpy as np 
from qiskit.primitives import Estimator 
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import I, X, Y, Z

H_A = I^I^Z^Z + I^I^X^I + I^I^I^X
H_B = Z^Z^I^I + X^I^I^I + I^X^I^I


E_A, J_A = np.linalg.eigh(H_A.to_matrix())
E_B, J_B = np.linalg.eigh(H_B.to_matrix())
sum(E_A == E_B),len(E_B)

optimizer = COBYLA(maxiter=100)

# gamma_1 = 0.1
# gamma_2 = 0.1
# alpha_1 = 0.1
# alpha_2 = 0.1
beta = 0
estimator = Estimator()
beta_coff = -pow(beta,-1.57) if beta != 0 else 0
observables = SparsePauliOp(["IIXI","IIIX","IXII","XIII","IIZZ", "ZZII","IXIX", "XIXI","IZIZ", "ZIZI"],
                            coeffs=[1,1,1,1,1.57,1.57,beta_coff,beta_coff,beta_coff,beta_coff])

def circuit(angles):
    gamma_1,gamma_2,alpha_1,alpha_2 = angles[0],angles[1],angles[2],angles[3]
    qc = qiskit.QuantumCircuit(4)

    qc.h(3)
    qc.h(2)
    qc.cnot(3,1)
    qc.cnot(2,0)

    labels_intra_2 = ["IIZZ", "ZZII"]
    coeffs_intra_2 = [1/2,1/2]
    spo_intra_2 = SparsePauliOp(labels_intra_2,coeffs_intra_2)

    labels_intra_1 = ["IIXI", "IIIX","XIII", "IXII"]
    coeffs_intra_1 = [1/2,1/2,1/2,1/2]
    spo_intra_1 = SparsePauliOp(labels_intra_1,coeffs_intra_1)

    labels_inter_1 = ["IXIX", "XIXI"]
    coeffs_inter_1 = [1/2,1/2]
    spo_inter_1 = SparsePauliOp(labels_inter_1,coeffs_inter_1)

    labels_inter_2 = ["IZIZ", "ZIZI"]
    coeffs_inter_2 = [1/2,1/2]
    spo_inter_2 = SparsePauliOp(labels_inter_2,coeffs_inter_2)

    qc.hamiltonian(spo_intra_1,gamma_1,list(range(4)))
    qc.hamiltonian(spo_intra_2,gamma_2,list(range(4)))
    qc.hamiltonian(spo_inter_1,alpha_1,list(range(4)))
    qc.hamiltonian(spo_inter_2,alpha_2,list(range(4)))

    job = estimator.run(qc,observables=observables)
    exp = job.result().values[0]
    return exp 

param = [0.1,0.3,0.2,0.3]
for _ in range(100):
    results = optimizer.minimize(circuit,param)
    param = results.x 
    print(results.fun)
    











    # qc.r(np.pi/2,np.pi/2+gamma_1,qubit=0)
    # qc.ry(np.pi/2,qubit=1)
    # qc.r(-np.pi/2,np.pi/2+gamma_1,qubit=2)
    # qc.r(-np.pi/2,np.pi/2+gamma_1,qubit=3)
    # qc.cz(0,2)
    # qc.cz(1,3)
    # qc.r(-np.pi/2,np.pi/2+gamma_1,qubit=0)
    # qc.rx(gamma_1,qubit=1)
    # qc.ry(np.pi/2,qubit=3)
    # qc.cz(0,1)
    # qc.cz(2,3)
    # qc.rx(gamma_2,qubit=0)
    # qc.rx(gamma_2,qubit=2)
    # qc.cz(0,1)
    # qc.cz(2,3)
    # qc.ry(np.pi/2,qubit=0)
    # qc.ry(-np.pi/2,qubit=3)
    # qc.cz(0,2)
    # qc.cz(1,3)
    # qc.rx(alpha_1,qubit=0)
    # qc.rx(alpha_1,qubit=1)
    # qc.rx(-alpha_2,qubit=2)
    # qc.rx(-alpha_2,qubit=3)
    # qc.cz(0,2)
    # qc.cz(1,3)
    # qc.ry(np.pi/2,qubit=2)
    # qc.ry(np.pi/2,qubit=3)