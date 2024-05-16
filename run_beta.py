import sys, qiskit
import qiskit.quantum_info
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
import qoop
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.core import ansatz, state, random_circuit
from qoop.backend import constant, utilities
from qoop.evolution import crossover, mutate, selection, threshold
from qoop.evolution.environment import EEnvironment, EEnvironmentMetadata
import pickle
import json
from qoop.compilation.qsp import QuantumStatePreparation, metric, QuantumCompilation
import glob 

filename = ""

def multiple_compile(num_qubitss,qcs,betas):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile,num_qubitss,qcs,betas)
    return results

def bypass_compile(num_qubits,qc,beta):
    return compiliation(num_qubits,qc,beta)

def compiliation(num_qubits,qc,beta):
    constant.PSI = state.construct_tfd_state(num_qubits, beta = beta,return_vector=True)
    qsp = QuantumCompilation(
            u=qc,
            vdagger = qiskit.QuantumCircuit(num_qubits*2,num_qubits*2)
        ).fast_fit(num_steps=500)
    gibbs_purity, gibbs_fidelity = metric.gibbs_metrics(u=qc,vdagger=constant.PSI.conj(),thetass=[qsp.thetas])
    # qsp.save(filename+f"/best_circuit_{beta}")
    return gibbs_purity, gibbs_fidelity 

def create_2qubit_circ(thetas):
    # from qiskit.circuit import ParameterVector
    # thetas = ParameterVector('theta',21*4)
    qc = qiskit.QuantumCircuit(4)
    qc.rx(thetas[0],2)
    qc.cry(thetas[1],1,3)
    qc.rx(thetas[2],0)
    qc.cx(1,2)
    qc.crx(thetas[3],0,3)
    qc.h(0)
    qc.h(1)
    qc.rx(thetas[4],2)
    qc.rx(thetas[5],3)
    qc.crz(thetas[6],0,1)
    qc.cx(2,3)
    qc.crx(thetas[7],1,3)
    qc.rx(thetas[8],2)
    qc.rz(thetas[9],0)
    qc.rz(thetas[10],1)
    qc.ry(thetas[11],3)
    qc.rx(thetas[12],0)
    qc.rz(thetas[13],2)
    qc.cry(thetas[14],0,3)
    qc.h(0)
    qc.cry(thetas[15],1,2)
    qc.crz(thetas[16],3,1)
    qc.ry(thetas[17],2)
    qc.rx(thetas[18],1)
    qc.ry(thetas[19],0)
    qc.cry(thetas[20],3,2)

    return qc 

def add_extra_rot_gates(qc,numqubit,thetas):
    for i in range(numqubit):
        qc.rx(thetas[i],i)
    for i in range(numqubit):
        qc.rz(thetas[i+numqubit],i)
    for i in range(numqubit):
        qc.rx(thetas[i+2*numqubit],i)
    return qc 

def create_block_3_qubits(thetas):
    qc = qiskit.QuantumCircuit(6)
    # extra_params = 18*3
    # orig_params = 21*3
    from qiskit.circuit import ParameterVector
    
    # thetas = ParameterVector('theta',orig_params)
    
    qc2b = create_2qubit_circ(thetas[:21])
    qc = qc.compose(qc2b,[0,1,3,4])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params:orig_params+18])

    
    qc2b = create_2qubit_circ(thetas[21:2*21])
    qc = qc.compose(qc2b,[1,2,4,5])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+18:orig_params+2*18])
    
    qc2b = create_2qubit_circ(thetas[2*21:3*21])
    qc = qc.compose(qc2b,[0,2,3,5])
    return qc
 
def train_bestc_circuit(filename):
    # qc2qbs = utilities.load_circuit(filename+"/best_circuit")

    thetas = None 

    num_qubits = 4


    # qc = qiskit.QuantumCircuit(6)
    # extra_params = 18*3
    # orig_params = 21*3
    # from qiskit.circuit import ParameterVector
    
    # thetas = ParameterVector('theta',orig_params+extra_params)
    # qc2b = create_2qubit_circ(thetas[:21])
    # qc = qc.compose(qc2b,[0,1,3,4])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params:orig_params+18])

    
    # qc2b = create_2qubit_circ(thetas[21:2*21])
    # qc = qc.compose(qc2b,[1,2,4,5])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+18:orig_params+2*18])
    
    # qc2b = create_2qubit_circ(thetas[2*21:3*21])
    # qc = qc.compose(qc2b,[0,2,3,5])
    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+2*18:orig_params+3*18])


    # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+2*18:orig_params+3*18])
    qc = qiskit.QuantumCircuit(8)
    orig_param = 3*4*21 
    from qiskit.circuit import ParameterVector
    thetas = ParameterVector('theta',orig_param)

    qc_3_1 = create_block_3_qubits(thetas[:3*21]) 
    qc = qc.compose(qc_3_1,[0,1,2,4,5,6])

    qc_3_2 = create_block_3_qubits(thetas[3*21:2*3*21])
    qc = qc.compose(qc_3_2,[1,2,3,5,6,7])
    
    qc_3_3 = create_block_3_qubits(thetas[2*3*21:3*3*21])
    qc = qc.compose(qc_3_3,[0,2,3,4,6,7])

    qc_3_4 = create_block_3_qubits(thetas[3*3*21:4*3*21])
    qc = qc.compose(qc_3_4,[0,1,3,4,5,7])
    
    print(qc)
    print(qc.depth())
    gibbs_purities, gibbs_fidelities = [], []
    betas = np.linspace(0,10,100)
    res = multiple_compile([num_qubits]*len(betas),[qc]*len(betas),betas)
    
    gibbs_purities = []
    gibbs_fidelities = []
    for r in res:
        gibbs_purities.append(r[0])
        gibbs_fidelities.append(r[1])

    print(gibbs_fidelities)
    print(gibbs_purities)

    
    print("End")
        


train_bestc_circuit("/home/viet/GA-QAS/4qubits_compilation_fitness_gibbs_2024-04-26")
# beta = np.linspace(0,10,100)
# res = []
# for b in beta:
#     psi = state.construct_tfd_state(4, beta = b,return_vector=True)
#     psi = np.expand_dims(psi,axis=0)
#     rho = psi.conj().T*psi
#     gibbs_sigma = qiskit.quantum_info.partial_trace(rho, [0, 1,2,3])
#     res.append(np.trace(np.linalg.matrix_power(gibbs_sigma, 2)))
# print(res)

# (0.06250000000000003+0j), (0.06771330733853156+0j), (0.08445778558545108+0j), (0.11456343218667628+0j), (0.15765933412703767+0j), (0.20981673068187412+0j), (0.2649830167857983+0j), (0.31756284053729633+0j), (0.36403546053872876+0j), (0.4030895373946566+0j), (0.4349479800932326+0j), (0.460601620552052+0j), (0.4812691655903131+0j), (0.49810764389455864+0j), (0.5120954726225081+0j), (0.5240094269394089+0j), (0.5344433839148754+0j), (0.5438406310160007+0j), (0.5525266686433912+0j), (0.560737525816196+0j), (0.5686424275420752+0j), (0.5763612235136013+0j), (0.5839774879324829+0j), (0.5915482404586769+0j), (0.5991111106929768+0j), (0.6066896021391113+0j), (0.6142969569696184+0j), (0.6219389957303081+0j), (0.6296162074631213+0j), (0.6373252916051917+0j), (0.6450602983268987+0j), (0.652813474017669+0j), (0.6605758895967077+0j), (0.6683379082789895+0j), (0.6760895341741908+0j), (0.6838206720335611+0j), (0.6915213204231846+0j), (0.6991817147522166+0j), (0.7067924323166603+0j), (0.7143444683979827+0j), (0.7218292901674944+0j), (0.7292388734652644+0j), (0.7365657262826217+0j), (0.7438029018607891+0j), (0.750944003638573+0j), (0.7579831837760187+0j), (0.7649151366027169+0j), (0.7717350880549607+0j), (0.7784387819508554+0j), (0.7850224637883971+0j), (0.7914828626252763+0j), (0.7978171715009147+0j), (0.8040230267838987+0j), (0.810098486766136+0j), (0.8160420097749849+0j), (0.8218524320334105+0j), (0.8275289454637618+0j), (0.8330710756016055+0j), (0.838478659761015+0j), (0.8437518255709486+0j), (0.8488909699833618+0j), (0.8538967388368413+0j), (0.8587700070446855+0j), (0.8635118594631043+0j), (0.8681235724833787+0j), (0.8726065963813201+0j), (0.8769625384480062+0j), (0.8811931469174619+0j), (0.885300295699609+0j), (0.8892859699203756+0j), (0.8931522522651753+0j), (0.8969013101171119+0j), (0.9005353834770249+0j), (0.9040567736489264+0j), (0.9074678326713579+0j), (0.9107709534727144+0j), (0.9139685607265756+0j), (0.917063102381479+0j), (0.9200570418383907+0j), (0.922952850748232+0j), (0.9257530024013085+0j), (0.928459965680146+0j), (0.9310761995472371+0j), (0.9336041480393039+0j), (0.9360462357400288+0j), (0.9384048637036941+0j), (0.9406824058027461+0j), (0.9428812054730031+0j), (0.9450035728310567+0j), (0.9470517821392348+0j), (0.9490280695944096+0j), (0.9509346314179159+0j), (0.9527736222247851+0j), (0.9545471536515071+0j), (0.9562572932225472+0j), (0.9579060634367937+0j), (0.9594954410561773+0j), (0.9610273565795919+0j), (0.9625036938862663+0j), (0.9639262900336394+0j)]

    # qc = qiskit.QuantumCircuit(6)
    # extra_params = 18*3
    # orig_params = 21*3
    # from qiskit.circuit import ParameterVector
    
    # thetas = ParameterVector('theta',orig_params)
    
    # qc2b = create_2qubit_circ(thetas[:21])
    # qc = qc.compose(qc2b,[0,1,3,4])
    # # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params:orig_params+18])

    
    # qc2b = create_2qubit_circ(thetas[21:2*21])
    # qc = qc.compose(qc2b,[1,2,4,5])
    # # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+18:orig_params+2*18])
    
    # qc2b = create_2qubit_circ(thetas[2*21:3*21])
    # qc = qc.compose(qc2b,[0,2,3,5])
    # # qc = add_extra_rot_gates(qc,num_qubits*2,thetas[orig_params+2*18:orig_params+3*18])
