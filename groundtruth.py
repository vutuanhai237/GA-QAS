import numpy as np 
import qiskit
def calculate_hamiltonian(num_qubits):
    # pauli_x = np.array([[0,1],[1,0]])
    # pauli_z = np.array([[1,0],[0,-1]])
    # identity = np.array([[1,0],[0,1]])
    if num_qubits == 1:
        labels = ["X","Z"]
        coeffs = [1]*2
    if num_qubits == 2:
        labels = ["XI", "IX","ZZ","ZZ"]
        coeffs = [1]*4
    if num_qubits == 3:
        labels = ["XII", "IXI","IIX","ZZI","IZZ","ZIZ"]
        coeffs = [1]*6
    if num_qubits == 4:
        labels = ["XIII", "IXII","IIXI","IIIX","ZZII","IZZI","IIZZ","ZIIZ"]
        coeffs = [1,1,1,1,1,1,1,1]
    if num_qubits == 5:
        labels = ["XIIII", "IXIII","IIXII","IIIXI","IIIIX","ZZIII","IZZII","IIZZI","IIIZZ","ZIIIZ"]
        coeffs = [1]*10
    if num_qubits == 6:
        labels = ["XIIIII", "IXIIII","IIXIII","IIIXII","IIIIXI","IIIIIX","ZZIIII","IZZIII","IIZZII","IIIZZI","IIIIZZ","ZIIIIZ"]
        coeffs = [1]*12
    spo = qiskit.quantum_info.SparsePauliOp(labels,coeffs)
    return spo.to_matrix() 


def find_eigenvec_eigenval(matrix):

    value, vector = np.linalg.eig(matrix)
    new_vector = []
    for v in range(0, len(vector)):
        holder = []
        for h in range(0, len(vector)):
            holder.append(vector[h][v])
        new_vector.append(holder)

    return [value, np.array(new_vector)]

def calculate_terms_partition(eigenvalues,beta):
    
    list_terms = []
    partition_sum = 0
    for i in eigenvalues:
        list_terms.append(np.exp(-0.5*beta*i))
        partition_sum = partition_sum + np.exp(-1*beta*i)

    return [list_terms, np.sqrt(float(partition_sum.real))]

num_qubits = 4
# 0 =  (0.25811002658190296-2.1190790274114865e-17j)
# 1 =  (0.5162902288565647-1.9404080358762862e-17j)
# 2 =  (0.5813670331515546+1.716235506122395e-33j)
# 3 =  (0.6037992247770967+4.645739726048193e-33j)
# 4 =  (0.6224511014501402-6.1197257103415095e-33j)
# 5 =  (0.6377985334448365-7.176466271391695e-34j)
for beta in range(11):

    matrix = calculate_hamiltonian(num_qubits)
    eigen = find_eigenvec_eigenval(matrix)
    partition = calculate_terms_partition(eigen[0],beta)

    vec = np.zeros(2**(2*num_qubits))

    for i in range(0, 2**num_qubits):
        
        addition = (float((partition[0][i]/partition[1]).real))*(np.kron(eigen[1][i], eigen[1][i]))
        vec = np.add(vec, addition)
    qc = qiskit.QuantumCircuit(num_qubits*2)
    amplitude_state = vec/np.sqrt(sum(np.absolute(vec) ** 2))
    qc.prepare_state(amplitude_state, list(range(0, num_qubits*2)))
    rho = qiskit.quantum_info.DensityMatrix(qc)
    rho = qiskit.quantum_info.partial_trace(rho, [0, 1])
    print(f"{beta} = ",np.trace(np.linalg.matrix_power(rho, 2)))