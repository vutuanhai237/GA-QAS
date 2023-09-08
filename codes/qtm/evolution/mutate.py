import qiskit
import random
import qtm.evolution
from .utils import compose_circuit

def bitflip_mutate(qc: qiskit.QuantumCircuit, pool, is_truncate=True):
    point = random.random()
    qc1, qc2 = qtm.evolution.divide_circuit(qc, point)
    qc1.barrier()
    qc21, qc22 = qtm.evolution.utils.divide_circuit_by_depth(qc2, 1)
    genome = qtm.random_circuit.generate_with_pool(qc.num_qubits, 1, pool)
    new_qc = compose_circuit([qc1, genome, qc22])
    if is_truncate:
        if new_qc.depth() > qc.depth():
            new_qc, _ = qtm.evolution.utils.divide_circuit_by_depth(
                new_qc, qc.depth())
    return new_qc
