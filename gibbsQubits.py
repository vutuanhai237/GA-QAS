import sys, qiskit
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import numpy as np
import qsee
<<<<<<< Updated upstream
from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state, random_circuit
from qsee.backend import constant, utilities
from qsee.evolution import crossover, mutate, selection, threshold
from qsee.evolution.environment import EEnvironment, EEnvironmentMetadata
=======
from qoop.compilation.qsp import QuantumStatePreparation
from qoop.core import ansatz, state, random_circuit
from qoop.backend import constant, utilities
from qoop.evolution import crossover, mutate, selection, threshold
from qoop.evolution.environment import EEnvironment, EEnvironmentMetadata
>>>>>>> Stashed changes
import pickle

#11
#223
#930
#4023
#16277
<<<<<<< Updated upstream
def superevol(num_qubits,depth,i=100):

    def compilation_fitness_more_than_2_gibbs(qc: qiskit.QuantumCircuit):
        qsp1 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
        ).fit(num_steps=10)
        
        #(max_fid1+max_fid2+max_fid3)
        return 1 - qsp1.compiler.metrics['loss_fubini_study'][-1]

    def compilation_fitness_gibbs(qc: qiskit.QuantumCircuit):
        qsp1 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
        ).fit(num_steps=10)
        
        qsp2 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 1).inverse()
        ).fit(num_steps=10)
                
        qsp3 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 5).inverse()
        ).fit(num_steps=10)
        
        qsp4 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 7).inverse()
        ).fit(num_steps=10)
        
        qsp5 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 10).inverse()
        ).fit(num_steps=10)
                
        #(max_fid1+max_fid2+max_fid3)
        return 1-np.average([qsp1.compiler.metrics['loss_fubini_study'][-1],
            qsp2.compiler.metrics['loss_fubini_study'][-1],qsp3.compiler.metrics['loss_fubini_study'][-1],
            qsp4.compiler.metrics['loss_fubini_study'][-1],
            qsp5.compiler.metrics['loss_fubini_study'][-1]]) 

    def full_compilation_fitness_gibbs(qc: qiskit.QuantumCircuit):
        #ve hinh density matrix cho n=2, toi beta=8,10, th khac: so sanh kq fid, purity:  0,1,2,3,4,5,6,7,8,9,10
        #co dinh beta=0, thay doi n, eps
         
        qsp1 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 0).inverse()
        ).fit(num_steps=100)        
        
        qsp2 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 1).inverse()
        ).fit(num_steps=100)
        
        qsp3 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 5).inverse()
        ).fit(num_steps=100)
        
        qsp4 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 7).inverse()
        ).fit(num_steps=100)
        
        qsp5 = QuantumStatePreparation(
            u=qc,
            target_state= state.construct_tfd_state(num_qubits, beta = 10).inverse()
        ).fit(num_steps=100)
                
        return 1-np.average([qsp1.compiler.metrics['loss_fubini_study'][-1],
            qsp2.compiler.metrics['loss_fubini_study'][-1],qsp3.compiler.metrics['loss_fubini_study'][-1],
            qsp4.compiler.metrics['loss_fubini_study'][-1],
            qsp5.compiler.metrics['loss_fubini_study'][-1]]) 
=======

def multiple_compile(num_qubitss,qcs,betas):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(bypass_compile,num_qubitss,qcs,betas)
    return results

def bypass_compile(num_qubits,qc,beta):
    return compiliation(num_qubits,qc,beta)

def compiliation(num_qubits,qc,beta):
    qsp = QuantumStatePreparation(
            u=qc,
            target_state = state.construct_tfd_state(num_qubits, beta = beta).inverse()
        ).fit(num_steps=30)
    return qsp.metrics['loss_fubini_study'][-1]

def superevol(num_qubits,depth):

    def compilation_fitness_gibbs(qc: qiskit.QuantumCircuit):
        betas = np.linspace(0,10,100)
        res = multiple_compile([num_qubits]*len(betas),[qc]*len(betas),betas)
        circs = []
        for r in res:
            circs.append(r)
        circs = np.array(circs)
        #weighted-sum
        a=2.2;b=1.6;c=0.9
        score = (a*np.average(circs[np.argwhere((betas<4))])+
                  b*np.average(circs[np.argwhere((4<=betas) & (betas<=7))])+c*np.average(circs[np.argwhere((7<=betas) & (betas<=10))]))/(a+b+c)
        return 1-score

>>>>>>> Stashed changes
            
    num_circuit = 16
    env_metadata = EEnvironmentMetadata(
        num_qubits = 2*num_qubits,
        depth = depth,
        num_circuit = num_circuit,
        num_generation = 20,
        prob_mutate=3/(depth * num_circuit)
    )
    env = EEnvironment(
        metadata = env_metadata,
        fitness_func= compilation_fitness_gibbs,
        selection_func=selection.elitist_selection,
        crossover_func=crossover.onepoint_crossover,
        mutate_func=mutate.layerflip_mutate,
        threshold_func=threshold.compilation_threshold
    )
<<<<<<< Updated upstream
    env.set_filename(f'Trial{i}')
    env.evol()

for i in range(100,130):
    superevol(2,9,i=i)
=======
    env.evol()

superevol(2,9)
>>>>>>> Stashed changes
