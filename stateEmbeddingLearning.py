import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_model import BasePolicy
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
import networkx as nx
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")


"""
--------------------------------------------------------------------------------------------------------------------
Algorithm: GATES
Paper: GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy. IJCAI 2025.
Authors: Ya Shen, Gang Chen, Hui Ma, and Mengjie Zhang
---------------------------------------------------------------------------------------------------------------------
Note: You need to adapt the algorithm to suit your specific problems and configurations to ensure fairness.
"""
class GATES(nn.Module):
    def __init__(self,
                 task_fea_size,
                 vm_fea_size,
                 output_size,
                 d_model,
                 att_heads,
                 att_en_layers,
                 d_ff,
                 gat_heads,
                 dropout=0.1):
        super(GATES, self).__init__()

        # Task preprocess
        self.task_feature_enhance = nn.Sequential(nn.Linear(task_fea_size, 128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, d_model))

        # VM preprocess
        self.vm_embedding = nn.Sequential(nn.Linear(vm_fea_size, d_model))

        # self-attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, att_heads, d_ff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, att_en_layers)

        # GATs
        self.gat_dag_layer01 = GATConv(6, d_model, heads=gat_heads, concat=True)
        self.gat_dag_layer02 = GATConv(gat_heads*d_model, d_model, heads=1, concat=False)
        self.gat_vm_layer01 = GATConv(4, d_model, heads=gat_heads, concat=True)
        self.gat_vm_layer02 = GATConv(gat_heads*d_model, d_model, heads=1, concat=False)


    def forward(self, device, ob, dag, node_id, VM_features_matrix):
        """
            ----------1)workflow_embedded + vm_global_info
        """
        ob = torch.from_numpy(ob.astype(np.float32)).to(device)
        VM_features_matrix = torch.from_numpy(VM_features_matrix.astype(np.float32)).to(device)

        readyTask_info = ob[0, 0:-4].unsqueeze(0)  
        vm_info = ob[:, -4::].unsqueeze(1)

        workflow_embedded = self.task_feature_enhance(readyTask_info)

        vm_embedded = self.vm_embedding(vm_info)
        vm_embedded = vm_embedded.permute(1, 0, 2)
        vm_global_info = self.transformer_encoder(vm_embedded)
        vm_global_info = vm_global_info.permute(1, 0, 2).squeeze(1)

        """
            ----------2)GAT for tasks DAG
        """
        adj_matrix = nx.adjacency_matrix(dag).todense()  
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

        node_features = []
        predecessor = adj_matrix.sum(dim=0)  
        successor = adj_matrix.sum(dim=1)   
        for node, data in dag.nodes(data=True):
            if node == node_id: 
                fea = [predecessor[node], successor[node], data["processTime"], data["size"], data["sub_deadline"], 2.0]
                node_features.append(fea)
            else:
                fea = [predecessor[node], successor[node], data["processTime"], data["size"], data["sub_deadline"], data["scheduled"]]
                node_features.append(fea)
        node_features = torch.tensor(node_features, dtype=torch.float32).to(device)

        readyTask_info_ = node_features[node_id, :].unsqueeze(0)
        readyTask_overall = torch.cat((readyTask_info, readyTask_info_), dim=-1).to(device)

        mean = node_features.mean(dim=0, keepdim=True)
        std = node_features.std(dim=0, keepdim=True)
        node_features = (node_features - mean) / std

        # Prepare the data for GAT_dag
        edge_index = adj_matrix.nonzero(as_tuple=False).t()                                
        edge_index_reversed = edge_index[[1, 0], :]                                        
        edge_index = torch.cat([edge_index, edge_index_reversed], dim=1).to(device) 
        data_for_GAT_dag = Data(x=node_features, edge_index=edge_index)

        # GAT_dag process the data
        dag_x, dag_edge_index = data_for_GAT_dag.x, data_for_GAT_dag.edge_index
        dag_x = F.elu(self.gat_dag_layer01(dag_x, dag_edge_index))
        dag_x = F.elu(self.gat_dag_layer02(dag_x, dag_edge_index))

        dag_x_mean = dag_x.mean(dim=0, keepdim=True)

        dag_x_ready = dag_x[node_id].unsqueeze(0)

        """
            ----------3)GAT for tasks-VM graph
        """
        readyTask_vm_features = torch.cat((readyTask_info, vm_info.squeeze(1)), dim=0).to(device)  
        num_nodes = readyTask_vm_features.shape[0]  
        target_nodes = torch.zeros(num_nodes, dtype=torch.long)
        neighbor_nodes = torch.arange(0, num_nodes, dtype=torch.long)
        edge_forward = torch.stack([target_nodes, neighbor_nodes], dim=0)               
        edge_backward = torch.stack([neighbor_nodes, target_nodes], dim=0)             
        taskVM_edge_index = torch.cat([edge_forward, edge_backward], dim=1).to(device)  

        data_for_GAT_vm = Data(x=readyTask_vm_features, edge_index=taskVM_edge_index)
        taskVM_x, taskVM_edge_index = data_for_GAT_vm.x, data_for_GAT_vm.edge_index
        taskVM_x = F.elu(self.gat_vm_layer01(taskVM_x, taskVM_edge_index))
        taskVM_x = F.elu(self.gat_vm_layer02(taskVM_x, taskVM_edge_index))
        taskVM_x_ready = taskVM_x[0].unsqueeze(0)

        """
            ----------4)----------
        """
        state_embedding = torch.cat((workflow_embedded, dag_x_mean, dag_x_ready, taskVM_x_ready), dim=-1)  
        state_embedding = state_embedding.expand(vm_global_info.shape[0], -1)
        state_embedding = torch.cat((state_embedding, taskVM_x[1:]), dim=-1)
        concatenation_embeddings = torch.cat((vm_global_info, state_embedding), dim=-1)

        return concatenation_embeddings, data_for_GAT_dag, readyTask_overall

