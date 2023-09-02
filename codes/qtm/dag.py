
import qiskit
import scipy
import qtm.constant
import numpy as np
import types
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)



def convert_string_to_int(string):
    return sum([ord(char) - 65 for char in string])


def circuit_to_dag(new_circuit, pnnp, preset, cset, qml_old_circuit):
    """Convert a circuit to graph.
    Read more: 
    - https://qiskit.org/documentation/retworkx/dev/tutorial/dags.html
    - https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commutation_dag.html

    Args:
        qc (qiskit.QuantumCircuit): A qiskit quantum circuit

    Returns:
        DAG: direct acyclic graph
    """
    return qml.transforms.commutation_dag(new_circuit)(pnnp,preset,cset,qml_old_circuit)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim = None, hidden_dim = None, output_dim = None, num_layers = None):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.layers.append(GraphConvolution(hidden_dim, output_dim))

    def forward(self, x, adj, fr_loss):
        x = F.relu(self.layers[0](x,adj)) 
        for layer in self.layers[1:]:
            x = F.relu(layer(x, adj))
        return x
    
    def dag_to_node_features(self,dag):
        node_features = []
        for i in range(dag.size):
            node = dag.get_node(0)
            operation = qtm.constant.look_up_operator[node.op.base_name]
            params = node.op.parameters
            if len(params) == 0:
                params = [0]
            node_features.append([qtm.dag.convert_string_to_int(operation), *params])
        return np.array(node_features)

    def dag_to_adjacency_matrix(self,dag):
        num_nodes = dag.size
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(dag.size):
            node = dag.get_node(i)
            for successor in node.successors:
                adjacency_matrix[i][successor] = 1
        return np.array(adjacency_matrix)


    def graph_to_scalar(self,node_features, adjacency_matrix, fr_loss):
        num_nodes = node_features.shape[0]
        # Forward pass through the model
        graph_embedding = self(node_features, adjacency_matrix, fr_loss)
        # Apply global sum pooling to obtain a scalar representation
        # Apply global sum pooling to obtain a scalar representation
        graph_scalar = torch.sum(graph_embedding)
        # Sigmod activation to ranged value from 0 to 1
        return torch.sigmoid(graph_scalar)

    def circuit_to_scalar(self, new_circuit, pnnp, preset, cset, qml_old_circuit,frozen_mean_cost, device: str)->float:
        """Evaluate circuit

        Args:
            qc (qiskit.QuantumCircuit): encoded circuit

        Returns:
            float: Value from 0 to 1
        """
        dag = qtm.dag.circuit_to_dag(new_circuit, pnnp, preset, cset, qml_old_circuit)
        node_features = torch.FloatTensor(self.dag_to_node_features(dag)).to(device)
        adjacency_matrix = torch.FloatTensor(self.dag_to_adjacency_matrix(dag)).to(device)
        return self.graph_to_scalar(node_features, adjacency_matrix,frozen_mean_cost)