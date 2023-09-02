
import qtm.qcompilation
import numpy as np
import types
import qiskit
import random_circuit
import matplotlib.pyplot as plt
import random_circuit
from qdas.DQASsearch import DQAS_search
from qdas.utils import set_op_pool
from qdas.vag import GHZ_vag 
# from pennylane import numpy as np1
import tensorflow as tf 
import qtm.constant
from functools import partial
import pennylane as qml
from qiskit import QuantumCircuit
from qdas.DQASsearch import get_preset, get_op_pool, get_weights
import qtm.dag
from qiskit.tools.visualization import circuit_drawer
import torch.distributions as distributions 
import torch 
import torch.optim as optim 
import random_circuit

ghz_pool = [
    ("RY", 0),
    ("RY", 1),
    ("RY", 2),
    ("CNOT", 0, 1),
    ("CNOT", 1, 0),
    ("CNOT", 0, 2),
    ("CNOT", 2, 0),
    ("Hadamard", 0),
    ("Hadamard", 1),
    ("Hadamard", 2)]

set_op_pool(ghz_pool)
c = len(ghz_pool)

num_qubits = 3
def LQcompilation(ansatz):
    theta = qiskit.circuit.ParameterVector('theta')
    thetas_length = 0
    qc = QuantumCircuit(num_qubits)
    thetas_length = 0
    parameters = []
    for ci in ansatz.data:
        if ci[0].name.startswith("measure"):
            continue
        if ci[0].name.startswith("r"):
            thetas_length += 1
            theta.resize(thetas_length)
            getattr(qc,ci[0].name)(theta[thetas_length-1],ci[1][0]._index)
            parameters.append(ci[0].params[0])
        elif ci[0].name.startswith("h"):
            getattr(qc,ci[0].name)(ci[1][0]._index)
        elif ci[0].name.startswith("cx"):
            getattr(qc,ci[0].name)(ci[1][0]._index,ci[1][1]._index)
    compiler = qtm.qcompilation.QuantumCompilation(
        u = qc,
        vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
        optimizer = 'adam',
        loss_func = 'loss_fubini_study',
        thetas = np.array(parameters)
    )
    compiler.fit(num_steps = 100, verbose = 1)
    return compiler.loss_values[-1], compiler.u.bind_parameters(compiler.thetas)

def reward_function(curr_objective_value,prev_objective_value,minimum_objective_value,step,max_layer,epsilon=1e-5):
    if curr_objective_value < epsilon:
        return 5
    elif step > max_layer:
        return -5
    else:
        return max((prev_objective_value-curr_objective_value)/(prev_objective_value-minimum_objective_value),-1)

def merge_circuit(qc1, qc2):
    return qc1.compose(qc2)

def calculate_returns(rewards,discounted_factor,normalize=True,device='cuda'):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discounted_factor
        returns.insert(0,R)
    
    returns = torch.FloatTensor(returns).to(device)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns 

def update_policy(returns,log_prob_actions,optimizer):
    returns = returns.detach()
    loss = - (returns * log_prob_actions).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def reset():
    qc = QuantumCircuit(3, 3)
    return qc 

def train(episode,nn_model,optimizer,num_steps,max_layer=2,optimal_max_step=2,discounted_factor = 0.99):
    # neural_network 
    initial_circuit = reset()
    nn_model.train()
    s = [] 
    rewards = [] 
    log_prob_actions = [] 
    dev_qiskit = qml.device("qiskit.aer", wires=3)
    s.append(initial_circuit)
    prev_objective_value = 10000 
    epsiode_reward = 0
    for t in range(0, num_steps):
        print(f"Step:{t}")
        old_circuit = s[t]
        qml_old_circuit =  qml.from_qiskit(old_circuit)
        GHZ_vag_func = partial(GHZ_vag,qml_old_circuit)
        stp, nnp, cset, hist, new_circuit =  DQAS_search(
            qml_old_circuit,
            GHZ_vag_func,
            nq=num_qubits,
            p=max_layer,
            batch=10,
            epochs=60,
            verbose=False,
            nnp_initial_value=np.zeros([max_layer, c]),
        )
        frozen_mean_cost = hist[-1]
        preset = get_preset(stp).numpy()
        pnnp = get_weights(nnp, stp).numpy()
        
        action_prob = nn_model.circuit_to_scalar(new_circuit, pnnp, preset, cset, qml_old_circuit,frozen_mean_cost, device = "cuda")
        dist = distributions.Bernoulli(action_prob)
        action  = dist.sample()
        print(f"Action:{action}")
        log_prob_actions.append(dist.log_prob(action))

        if action == 0:
            s.append(old_circuit)
        else:
            qcir = qml.QNode(new_circuit, dev_qiskit, interface="torch") 
            qcir(pnnp,preset,cset,qml_old_circuit,rqiskit=True)
            new_qcircuit = dev_qiskit._circuit
            s.append(new_qcircuit)
        
        # circuit_drawer(s[t+1],output='mpl',filename=f"/home/viet/EvolutionalQuantumCircuit/codes/evocirc/qiskitcircEp:{episode}_{t}.png")
        
        curr_objective_value, compiled_u = LQcompilation(s[t+1])
        print(curr_objective_value)
        s[t+1] = compiled_u
        print("Current compiled loss",compiled_u)
        reward = reward_function(curr_objective_value,prev_objective_value,0,t,optimal_max_step)
        epsiode_reward += reward
        rewards.append(reward)
    log_prob_actions = torch.stack(log_prob_actions)
    returns = calculate_returns(rewards,discounted_factor)
    loss_pg = update_policy(returns,log_prob_actions,optimizer)
    return loss_pg,epsiode_reward

train_rewards = []
MAX_EPISODES = 40
num_steps = 3
hidden_dim = 64
output_dim = 1
input_dim = 2
num_layers = 2
nn_model = qtm.dag.GCN(input_dim= 2 ,hidden_dim = hidden_dim, output_dim = output_dim, num_layers = num_layers).cuda()
learning_rate = 0.01
optimizer = optim.Adam(nn_model.parameters(),lr=learning_rate)

for episode in range(1, MAX_EPISODES+1):

    loss, train_reward = train(episode,nn_model,optimizer,num_steps=5)
    train_rewards.append(train_reward)
    mean_train_rewards = np.mean(train_rewards[-6:])

    if episode % 3: 
        print(f'| Episode:{episode}| Mean train rewards: {mean_train_rewards}|')

import pickle 

with open("/home/viet/EvolutionalQuantumCircuit/codes/train_rewards.pkl","wb") as handle:
    pickle.dump(train_rewards,handle)