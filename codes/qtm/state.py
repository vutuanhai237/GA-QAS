import qiskit, numpy as np
import qtm.gate, qtm.encoding
from qiskit.quantum_info import SparsePauliOp

def w(qc: qiskit.QuantumCircuit, num_qubits: int, shift: int = 0):
    """The below codes is implemented from [this paper](https://arxiv.org/abs/1606.09290)
    \n Simplest case: 3 qubits. <img src='../images/general_w.png' width = 500px/>
    \n General case: more qubits. <img src='../images/general_w2.png' width = 500px/>

    Args:
        - num_qubits (int): number of qubits
        - shift (int, optional): begin wire. Defaults to 0.

    Raises:
        - ValueError: When the number of qubits is not valid

    Returns:
        - qiskit.QuantumCircuit
    """
    if num_qubits < 2:
        raise ValueError('W state must has at least 2-qubit')
    if num_qubits == 2:
        # |W> state ~ |+> state
        qc.h(0)
        return qc
    if num_qubits == 3:
        # Return the base function
        qc.w3(shift)
        return qc
    else:
        # Theta value of F gate base on the circuit that it acts on
        theta = np.arccos(1 / np.sqrt(qc.num_qubits - shift))
        qc.cf(theta, shift, shift + 1)
        # Recursion until the number of qubits equal 3
        w(qc, num_qubits - 1, qc.num_qubits - (num_qubits - 1))
        for i in range(1, num_qubits):
            qc.cnot(i + shift, shift)
    return qc


def create_ghz_state(num_qubits, theta: float = np.pi / 2):
    """Create GHZ state with a parameter

    Args:
        - num_qubits (int): number of qubits
        - theta (float): parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.ry(theta, 0)
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(0, i + 1)
    return qc


def create_ghz_state_inverse(num_qubits: int, theta: float = np.pi / 2):
    """Create GHZ state with a parameter

    Args:
        - qc (QuantumCircuit): Init circuit
        - theta (Float): Parameter

    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(theta, float) != True:
        theta = (theta['theta'])
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for i in range(0, num_qubits - 1):
        qc.cnot(0, num_qubits - i - 1)
    qc.ry(-theta, 0)
    return qc

def create_haar_state(num_qubits: int):
    """Create a random Haar quantum state

    Args:
        num_qubits (int): number of qubits

    Returns:
        qiskit.QuantumCircuit
    """
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.prepare_state(psi, list(range(0, num_qubits)))
    return qc

def create_haar_state_inverse(num_qubits: int):
    psi = 2*np.random.rand(2**num_qubits)-1
    psi = psi / np.linalg.norm(psi)
    encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')
    qc = encoder.qcircuit
    return qc.inverse()

def create_w_state(num_qubits):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)
    qc = w(qc, qc.num_qubits)
    return qc


def create_w_state_inverse(qc: qiskit.QuantumCircuit):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc1 = qiskit.QuantumCircuit(qc.num_qubits)
    qc1.x(0)
    qc1 = w(qc1, qc.num_qubits)
    qc = qc.combine(qc1.inverse())
    return qc

def create_ame_state(num_qubits: int):
    if num_qubits == 3:
        amplitude_state = np.array([
                0.27,
                0.363,
                0.326,
                0,
                0.377,
                0,
                0,
                0.740*(np.cos(-0.79*np.pi)+1j*np.sin(-0.79*np.pi))])
    if num_qubits == 4:
        w = np.exp(2*np.pi*1j/3)
        amplitude_state = 1/np.sqrt(6)*np.array([
                0.0,
                0.0,
                1,
                0,
                0,
                w,
                w**2,
                0,
                0,
                w**2,
                w,
                0,
                1,
                0,
                0,
                0,])
    amplitude_state = amplitude_state/np.sqrt(sum(np.absolute(amplitude_state) ** 2))
    qc = qiskit.QuantumCircuit(num_qubits,num_qubits)
    qc.prepare_state(amplitude_state, list(range(0, num_qubits)))
    return qc


def create_ame_state_fake(num_qubits: int):
    amplitude_state = np.array([
            0.27,
            0.363,
            0.326,
            0,
            0.377,
            0,
            0,
            np.sqrt(1-0.27**2-0.363**2-0.326**2-0.377**2)])
    qc = qiskit.QuantumCircuit(num_qubits,num_qubits)
    qc.prepare_state(amplitude_state, [0,1,2])
    return qc

def create_ame_state_inverse(qc: qiskit.QuantumCircuit):
    """Create n-qubit W state based on the its number of qubits

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    qc1 = create_ame_state(qc.num_qubits)
    qc = qc.combine(qc1.inverse())
    return qc


def calculate_hamiltonian(num_qubits):
    # pauli_x = np.array([[0,1],[1,0]])
    # pauli_z = np.array([[1,0],[0,-1]])
    # identity = np.array([[1,0],[0,1]])
    labels = ["XI", "IX","ZZ","ZZ"]
    coeffs = [1,1,1,1]
    spo = SparsePauliOp(labels,coeffs)
    return spo.to_matrix() 

# Calculating the eigenvalues and eigenvectors of the cost Hamiltonian

def find_eigenvec_eigenval(matrix):

    value, vector = np.linalg.eig(matrix)
    new_vector = []
    for v in range(0, len(vector)):
        holder = []
        for h in range(0, len(vector)):
            holder.append(vector[h][v])
        new_vector.append(holder)

    return [value, np.array(new_vector)]

# Preaparing the partition function and each of the probability amplitudes of the diifferent terms in the TFD state

def calculate_terms_partition(eigenvalues,beta):

    list_terms = []
    partition_sum = 0
    for i in eigenvalues:
        list_terms.append(np.exp(-0.5*beta*i))
        partition_sum = partition_sum + np.exp(-1*beta*i)

    return [list_terms, np.sqrt(float(partition_sum.real))]

#Preparring the TFD state for the cost function

def construct_tfd_state(num_qubits,beta):

    # In this implementation, the eigenvectors of the Hamiltonian and the transposed Hamiltonian are calculated separately
    y_gate = 1
    y = np.array([[0, -1], [1, 0]])
    for i in range(0, num_qubits):
        y_gate = np.kron(y_gate, y)
    
    matrix = calculate_hamiltonian(num_qubits)
    eigen = find_eigenvec_eigenval(matrix)
    partition = calculate_terms_partition(eigen[0],beta)

    vec = np.zeros(2**(2*num_qubits))
    for i in range(0, 2**num_qubits):
        
        # time_rev = complex(0,1)*np.matmul(y_gate, np.conj(eigen[1][i]))
        addition = (float((partition[0][i]/partition[1]).real))*(np.kron(eigen[1][i],eigen[1][i]))
        vec = np.add(vec, addition)

    qc = qiskit.QuantumCircuit(4,4)
    amplitude_state = vec/np.sqrt(sum(np.absolute(vec) ** 2))
    qc.prepare_state(amplitude_state, list(range(0, num_qubits**2)))
    # rho = qiskit.quantum_info.DensityMatrix.from_instruction(qc)
    # gibbs_rho = qiskit.quantum_info.partial_trace(rho, [0, 1]).to_statevector()
    # qc = qiskit.QuantumCircuit(2,2)
    # qc.prepare_state(gibbs_rho, list(range(0, 2**2)))
    return qc
