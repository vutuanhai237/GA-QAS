import sys
import matplotlib.pyplot as plt
import qiskit
from qsee.backend import constant
from qsee.compilation.qsp import QuantumStatePreparation, metric
from qsee.backend import utilities
from qsee.core import state
import json 
import torch 
import torch.nn as nn  
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit.primitives import Sampler
import numpy as np 
import time
from einops import rearrange 
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import random 

num_qubits = 3

operations = {
    "h":1,
    "cx":2,
    "rx":3,
    "ry":4,
    "rz":5,
    "crx":6,
    "cry":7,
    "crz":8
}
vocab_size = 9 

param_operations = ["rx",
                    "ry",
                    "rz",
                    "crx",
                    "cry",
                    "crz"]

class GibbsDataset(Dataset):
    def __init__(self,file_list,num_qubits,device):
        self.device = device 
        self.list_qc = []
        self.num_qubits = num_qubits
        self.file_list = file_list 
        for file in file_list:
            self.list_qc.append(utilities.load_circuit(file))

        self.list_vdagger = []
        for _ in self.list_qc:
            list_vdagger_each_qc = []
            for i in range(0,11):
                list_vdagger_each_qc.append(state.construct_tfd_state(num_qubits=num_qubits,beta=i).inverse())
            self.list_vdagger.append(list_vdagger_each_qc)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        qc = self.list_qc[index]
        qcc = qc.copy()
        vdaggers = self.list_vdagger[index]
        composed_qcs = []
        for i in range(len(vdaggers)):
            composed_qcs.append([qcc.compose(vdaggers[i]),qcc,vdaggers[i]])
        parameter_orders = torch.zeros(2*self.num_qubits,qc.depth())-1
        arch = torch.zeros(2*self.num_qubits,qc.depth())
        for qubit in range(0,2*self.num_qubits):
            found = 0
            for i in range(0,len(qc._data)):
                if qubit == qc.find_bit(qc._data[i].qubits[0])[0]:
                    arch[qubit,found] = operations[qc._data[i][0].name]
                    if qc._data[i][0].name in param_operations:
                        parameter_orders[qubit,found] = qc._data[i][0].params[0]._index
                    found += 1 
        arch = arch.view(2*self.num_qubits*qc.depth()).long()
        attn_mask = (parameter_orders == -1).view(2*self.num_qubits*qc.depth())
        segment = (arch != 0).long()
        return arch.to(self.device),attn_mask.to(self.device),segment.to(self.device),parameter_orders.to(self.device),composed_qcs

def collate_fn(data):
    arch_tensor = []
    attn_mask_tensor = []
    segment_tensor = []
    parameter_orders = []
    composed_qcs = []
    for d in data:
        arch_tensor.append(d[0])
        attn_mask_tensor.append(d[1])
        segment_tensor.append(d[2])
        parameter_orders.append(d[3])
        composed_qcs.append(d[4])
    return torch.stack(arch_tensor),torch.stack(attn_mask_tensor),torch.stack(segment_tensor),torch.stack(parameter_orders),composed_qcs

file_list = ["/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_1",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_2",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_3",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_4",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_5",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_6",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_7",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_8",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_9",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_10",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_11",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_12",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_13",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_14",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_15",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_16",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_17",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_18",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_19",
            "/home/fptu/vJ3XOz68/qhack/GA-QAS/Trial_2_6qubits_compilation_fitness_more_than_2_gibbs_2024-01-23/best_circuit_20"]

# for qubit in range(0,6):
#     found = 0
#     for i in range(0,len(qc._data)):
#         if qubit == qc.find_bit(qc._data[i].qubits[0])[0] :
#             arch[qubit,found] = operations[qc._data[i][0].name]
#             if qc._data[i][0].name in param_operations:
#                 parameter_orders[qubit,found] = qc._data[i][0].params[0]._index
#             found += 1 

# q,l = arch.size()
# arch = arch.view(q*l).long()
# print(arch)
# segment = (arch != 0).long()

# attn_mask = (parameter_orders == -1).view(1,q*l)
# pad_attn_mask = pad_attn_mask.data.unsqueeze(1)
# pad_attn_mask = pad_attn_mask.expand(1, q*l, q*l)

class Embedding(nn.Module):
   def __init__(self,d_model):
       super(Embedding, self).__init__()
       self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
       self.seg_embed = nn.Embedding(2, d_model)  # segment(token type) embedding
       self.norm = nn.LayerNorm(d_model)

   def forward(self, x, seg):
       embedding = self.tok_embed(x) + self.seg_embed(seg)
       return self.norm(embedding)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=768, d_k=768, d_v=768, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-12)

    def forward(self, q, k, v, mask=None):
        residual = q
        q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask[None,:,None,:], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention()  # batch_size x sentence size x dim_inp

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        output, attn = self.attention(input_tensor,input_tensor,input_tensor, attention_mask)
        return output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Model(nn.Module):
    def __init__(self,num_layer=3,d_model=768):
        super().__init__()
        self.num_heads = 8
        self.embedding = Embedding(d_model)
        self.modules = []
        for i in range(num_layer):
            self.modules.append(Encoder())
        self.transformers_block = nn.Sequential(*self.modules)
        self.bertPreTransform = BertPredictionHeadTransform(hidden_size=768)
        self.reduced_head = nn.Linear(d_model, 11)
        self.activ = nn.GELU()
        # self.norm = nn.LayerNorm(q*l)

    def forward(self,arch,segment,enc_self_attn_mask,parameter_orders,qc):
        b,q,l = parameter_orders.size()
        output = self.embedding(arch, segment)
        # enc_self_attn_mask_transformers = enc_self_attn_mask.tile(batch_size*self.num_heads,1,1)
        for layer in self.transformers_block:
           output = layer(output, enc_self_attn_mask)
        out = self.reduced_head(self.bertPreTransform(output))
        out = out.view(b,q,l,11)
        # out = self.norm(self.activ(self.linear(output)).squeeze(-1)).squeeze(0).view(q,l)
        tloss = 0
        total_normal_params = []
        start = time.time()
        for ind in range(b):
            ind_normal_params = []
            total_param_ops = torch.sum(torch.where(parameter_orders[ind] != -1 ,1,0),axis=0).sum()
            random_samples = random.sample(range(0, 11), 3)
            random_circuits = []
            for i in random_samples:
                random_circuits.append(composeqcs[ind][i])
            for beta_order,(qc,_,_) in enumerate(random_circuits):
                params = []
                normal_params = []
                for i in range(0, total_param_ops):
                    x,y = torch.where(parameter_orders[ind].squeeze(0) == i)
                    params.append(out[ind,x,y,beta_order])
                    normal_params.append(out[ind,x,y,beta_order].detach().item())
                qnn = SamplerQNN(circuit=qc,weight_params=params,sampler=Sampler())
                params = torch.cat(params)
                tqc = TorchConnector(qnn,initial_weights=params)
                # tqc.training = False
                # for p in tqc.parameters():
                #     p.requires_grad = False
                results = tqc()
                qloss = 1-results[0]
                tloss += qloss 
                ind_normal_params.append(normal_params)
            total_normal_params.append(ind_normal_params)
        print("simulate time:",time.time()-start)
        return tloss/(3*b),total_normal_params

gibbsDataset = GibbsDataset(file_list,num_qubits,"cuda")
loader = DataLoader(gibbsDataset,batch_size=5,collate_fn=collate_fn)

model = Model().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



for epoch in range(20):
    for batch in tqdm(loader):
        optimizer.zero_grad()
        start = time.time()
        arch_tensor,attn_mask_tensor,segment_tensor,parameter_orders,composeqcs = batch
        loss,p = model(arch_tensor,segment_tensor, attn_mask_tensor,parameter_orders,composeqcs)
        end = time.time() - start
        print("first:",end)
        print("loss:",loss)
        loss.backward()
        # for ind in range(len(batch[-1])):
        #     for i,(_,u,vdagger) in enumerate(composeqcs[ind]):
        #         gibbs_purity, gibbs_fidelity = metric.gibbs_metrics(u=u,vdagger=vdagger,thetass=[p[ind][i]])
        #         print(f"individualth in batch:{ind}, beta:{i}, gibbs_purity:{np.real(gibbs_purity)},gibbs_fidelity: {np.real(gibbs_fidelity)}")
        # start = time.time()
        # end = time.time() - start
        # print("second:",end)
        print(f"Loss: {loss.item()}")
        optimizer.step()
        scheduler.step()

