from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
)
Tensor = Any 
from pennylane import numpy as np
from qdas.utils import get_op_pool,set_op_pool
import pennylane as qml 
import tensorflow as tf
from qdas.backends import get_backend
import torch 
import qiskit 
import qtm 

npdtype = np.complex64
backend = get_backend("tensorflow")

def array_to_tensor(*num: np.array) -> Any:
    l = [backend.convert_to_tensor(n.astype(npdtype)) for n in num]
    if len(l) == 1:
        return l[0]
    return l

def circuit(pnnp,preset,cset,old_circuit,rqiskit=False):
    if rqiskit:
        old_circuit(wires=[0,1,2])
    else:
        old_circuit(wires=[2,1,0])

    for i,j in enumerate(preset):
        gate = cset[j]
        if gate[0].startswith('R'):
            if rqiskit:
                getattr(qml,gate[0])(pnnp[i],3-1-gate[1])
            else:
                getattr(qml,gate[0])(pnnp[i],gate[1])
        elif gate[0] == 'Hadamard':
            if rqiskit:
                getattr(qml,gate[0])(3-1-gate[1])
            else:
                getattr(qml,gate[0])(gate[1])
        elif gate[0] == "CNOT":
            if rqiskit:
                qml.CNOT(wires=(3-1-gate[1],3 - 1- gate[2]))
            else:
                qml.CNOT(wires=(gate[1], gate[2]))
        elif gate[0] == "Identity":
            continue
    if rqiskit:
        return qml.expval(qml.PauliZ(0))
    return  qml.state()
    
def GHZ_vag(old_circuit, gdata: Tensor,nnp: Tensor, preset: Sequence[int]
            , verbose: bool = False, n: int = 3) -> Tuple[Tensor, Tensor]:
    reference_state = np.zeros([2**n])
    reference_state[0] = 1 / np.sqrt(2)
    reference_state[-1] = 1 / np.sqrt(2)
    
    nnp = nnp.numpy() 
    pnnp = [nnp[i,j] for i,j in enumerate(preset)]
    pnnp = np.array(pnnp)
    dev = qml.device("default.qubit", wires=n)
    
    cset = get_op_pool()
    pnnp = torch.tensor(pnnp)
    pnnp.requires_grad_()
    cir = qml.QNode(circuit, dev, interface="torch") 

    s = cir(pnnp,preset,cset,old_circuit)
    reference_state = torch.tensor(reference_state).clone().detach()
    loss = torch.sum(torch.abs(s - reference_state))
    try:
        loss.backward()
        gr = pnnp.grad
    except:
        gr = torch.zeros_like(pnnp)

    # gr = backend.real(gr)
    # gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
    gmatrix = np.zeros_like(nnp)
    for i, j in enumerate(preset):
        gmatrix[i, j] = gr[i]
    gmatrix = torch.FloatTensor(gmatrix)
    return loss, gmatrix,circuit
    
if __name__ == '__main__':
    p=4
    ghz_pool = [
    ("RY", 0),
    ("RY", 1),
    ("RY", 2),
    ("CNOT", 0, 1),
    ("CNOT", 1, 0),
    ("CNOT", 0, 2),
    ("CNOT", 2, 0),
    ("H", 0),
    ("H", 1),
    ("H", 2),]
    c = len(ghz_pool)
    set_op_pool(ghz_pool)
    nnp = tf.Variable(np.random.uniform(size=[p, c]))
    preset = tf.constant([0,5,3,1])
    a,b = GHZ_vag(nnp,preset)
    print(a,b)