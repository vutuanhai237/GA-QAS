import numpy as np
import qiskit.quantum_info as qi

def create_params(depths, num_circuits, num_generations):
    # zip num_generations, depths, num_circuits as params
    params = []
    for num_generation in num_generations:
        for depth in depths:
            for num_circuit in num_circuits:
                params.append((depth, num_circuit, num_generation))
    return params        

def calculate_risk(utests, V):
    num_qubits = V.num_qubits
    # Create |0> state
    zero_state = np.zeros((2**num_qubits, 1))
    zero_state[0] = 1
    # Create |0><0| matrix
    zero_zero_dagger = np.outer(zero_state, np.conj(zero_state.T))
    V_matrix = qi.DensityMatrix(V).data
    risk = []
    for utest in utests:
        Ui_matrix = qi.DensityMatrix(utest).data
        # Eq inside L1 norm of matrix ^2
        eq = (Ui_matrix @ zero_zero_dagger @ np.conj(Ui_matrix.T) - V_matrix @ zero_zero_dagger @ np.conj(V_matrix.T))
        # L1 norm of matrix ^ 2
        risk.append(np.linalg.norm(eq, 1)**2)
    # Expected risk / 4
    return np.mean(risk)/4  