
"""
    Algorithm: DEFT
    Paper Title: Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts
    Author: Ya Shen, Gang Chen, Hui Ma, Mengjie Zhang
    Conference: The Fourteenth International Conference on Learning Representations (ICLR 2026)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from policy.base_model import BasePolicy
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx
import warnings
warnings.filterwarnings("ignore", message=".*flash attention.*")
from DEFT.stateEmbeddingLearning import GATES
from DEFT.moe import HybridMoE, DAGEncoder 


class WFPolicy(BasePolicy):
    def __init__(self, config, policy_id=-1):
        super(WFPolicy, self).__init__()
        self.config = config 
        self.policy_id = policy_id  
        self.state_num = config['state_num']
        self.action_num = config['action_num']
        self.discrete_action = config['discrete_action']
        if "add_gru" in config:
            self.add_gru = config['add_gru']
        else:
            self.add_gru = True

        # numbert of model_paths = number of experts in MoE
        # You need to specify the pre-trained model paths of PMM in GATES and load the weights for initializing the MoE experts
        model_paths = [
            r'./...',
            r'./...',
            r'./...',
            r'./...',
        ]
        
        # SEM from GATES: for state embedding learning
        self.model = GATES(task_fea_size=4, vm_fea_size=4, output_size=1, d_model=16,
                           att_heads=2, att_en_layers=2, d_ff=128, gat_heads=2, dropout=0.1)

        self.dag_emb = DAGEncoder(node_dim=6, hidden_dim=32, dag_dim=16, num_heads=2)

        # MoE networks: for expert selection
        self.moe = HybridMoE(action_dim=96,
                             dag_dim=16,
                             ready_task_dim=4+6,
                             sla_gamma_dim=1, 
                             num_experts=4,
                             top_k=1,
                             expert_hidden_dim=128,
                             mode_source="parallel",  # default parallel in DEFT
                             shared_expert=True,
                             gating_nn_type="attn",  # graph-adaptive gating network
                             post_train_exp=True,  # False: keep the pre-trained PMM experts frozen (sometimes could be better for performance)
                             model_paths=model_paths,
                             model_name="GATES",  # the backbone of DEFT comes from GATES
                             )

    def forward(self, device, ob, dag, node_id, VM_features_matrix, sla_gamma, removeVM=None):
        # self.model.to(device)
        # self.dag_emb.to(device)
        # self.moe.to(device)

        concatenation_embeddings, data_for_GAT_dag, ready_task = self.model(device, ob, dag, node_id, VM_features_matrix)
        dag_embedding = self.dag_emb(x=data_for_GAT_dag.x, edge_index=data_for_GAT_dag.edge_index, batch=None)
        sla_gamma = torch.tensor(sla_gamma, dtype=torch.float32).view(1, 1).to(device)
        out = self.moe(concatenation_embeddings, dag_embedding, ready_task, sla_gamma)  

        logits = out.squeeze().to(device)
        if logits.dim() != 1:
            logits = logits.view(-1)

        if removeVM is not None:
            idx = torch.as_tensor(list(removeVM) if isinstance(removeVM, (list, tuple, set, np.ndarray, torch.Tensor)) else [removeVM], device=device, dtype=torch.long)
            logits[idx] = float("-inf")

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
        if torch.isinf(logits).all() and (logits < 0).all():
            logits = torch.zeros_like(logits)  

        if self.discrete_action:
            if self.config['greedy_action']:
                action = torch.argmax(logits)
            else:
                with torch.amp.autocast('cuda', enabled=False):
                    dist = torch.distributions.Categorical(logits=logits.float())
                    action = dist.sample()
        else:
            action = torch.relu(logits)

        action_np = action.detach().cpu().numpy()
        out_1xA = logits.view(1, -1) 
        return action_np, out_1xA

    def xavier_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0) 

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def norm_init(self, std=1.0):
        for param in self.parameters():
            shape = param.shape
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            param.data = torch.from_numpy(out)

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id

    def reset(self):
        pass

    def get_param_list(self):
        param_lst = []
        for param in self.parameters():
            param_lst.append(param.data.numpy())
        return param_lst

    def set_param_list(self, param_lst: list):
        lst_idx = 0
        for param in self.parameters():
            param.data = torch.tensor(param_lst[lst_idx]).float()
            lst_idx += 1

