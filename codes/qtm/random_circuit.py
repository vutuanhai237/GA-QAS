import numpy as np
import qiskit
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Reset
from qiskit.circuit.exceptions import CircuitError
import qtm.constant
import random


def random_choice_with_condition(d, k):
    item = None
    while item is None:
        item = random.choice(d)
        if item <= k:
            return item
        else:
            item = None
    return item


def random_circuit2(num_qubits, depth, pool=qtm.constant.operations, max_operands=3, measure=False,
                    conditional=False, reset=False, seed=None):

    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(num_qubits, num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(seed)
    thetas = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0
    for _ in range(depth):
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = random_choice_with_condition([1,1,1,1,1,1,1,1,2,2], max_possible_operands)
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            num_op_pool = [item for item in pool if item['num_op'] == num_operands]
            
            operation = rng.choice(num_op_pool)
            num_params = operation['num_params']
            thetas_length += num_params
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_params:thetas_length]
            register_operands = [qr[i] for i in operands]
            op = operation['operation'](*angles)
            qc.append(op, register_operands)
    return qc


def random_circuit(num_qubits, depth, max_operands=3, measure=False,
                   conditional=False, reset=False, seed=None):

    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(num_qubits, num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    rng = np.random.default_rng(seed)
    thetas = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            # num_operands = random.choice([1,1,1,1,1,2,2,2,2,2,3])
            # print(num_operands)
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            if num_operands == 1:
                operation = rng.choice(qtm.constant.one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(qtm.constant.two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(qtm.constant.three_q_ops)
            if operation in qtm.constant.one_param:
                num_angles = 1
            elif operation in qtm.constant.two_param:
                num_angles = 2
            elif operation in qtm.constant.three_param:
                num_angles = 3
            else:
                num_angles = 0
            # angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            thetas_length += num_angles
            thetas.resize(thetas_length)
            angles = thetas[thetas_length - num_angles:thetas_length]

            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)
            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc


def random_nonparam_circuit(num_qubits, depth, max_operands=3, measure=False,
                            conditional=False, reset=False, seed=None):

    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")

    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # apply arbitrary random operations at every depth
    for _ in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [
                q for q in remaining_qubits if q not in operands]
            if num_operands == 1:
                operation = rng.choice(qtm.constant.one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(qtm.constant.two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(qtm.constant.three_q_ops)
            if operation in qtm.constant.one_param:
                num_angles = 1
            elif operation in qtm.constant.two_param:
                num_angles = 2
            elif operation in qtm.constant.three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc
