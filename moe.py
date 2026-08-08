
"""
Mixture of Experts (MoE) with Graph-adaptive Gating for Dynamic Workflow Scheduling
Note: You need to adapt the algorithm to suit your specific problems and configurations to ensure fairness.
"""

from __future__ import annotations

import math
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, GCNConv


class DAGEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        dag_dim: int,
        *,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True)
        self.gat_ro = nn.Linear(hidden_dim * num_heads, dag_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_gat = F.elu(self.gat1(x, edge_index))
        x_gat = F.elu(self.gat2(x_gat, edge_index))
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long) 
        gat_pool = global_mean_pool(x_gat, batch)  
        dag_emb = F.elu(self.gat_ro(gat_pool))     

        return dag_emb


# ---------------------------------------------------------------------------
# 2) Graph-adaptive gating network
# might not be the best implementation according to the original paper
# you can re-implement it by yourself
# ---------------------------------------------------------------------------
class HybridRoutingGating(nn.Module):

    def __init__(
        self,
        action_dim: int,
        dag_dim: int,
        ready_task_dim: int,
        sla_gamma_dim: int,
        *,
        d_model: int = 128,
        num_experts: int = 4,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        token = torch.randn(2, num_experts, d_model) / math.sqrt(d_model)
        self.expert_token = nn.Parameter(token)  # learnable
        self.q_proj = nn.Linear(action_dim + dag_dim + ready_task_dim + sla_gamma_dim, d_model)  # Query projection (K,V are expert tokens)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.out_mode = nn.Linear(d_model, 2)

    def _attend_expert_tokens(self, q: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        _, attn_w = self.attn(q, token, token) 
        return attn_w.squeeze(1) * (self.d_model ** 0.5)

    def forward(
        self,
        actions: torch.Tensor,   
        dag: torch.Tensor,       
        ready: torch.Tensor,     
        sla_gamma: torch.Tensor 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = actions.size(0)
        q_ctx = torch.cat([actions, dag.expand(N, -1), ready.expand(N, -1), sla_gamma.expand(N, -1)], dim=-1)  # query
        q = self.q_proj(q_ctx).unsqueeze(1)   
        mode_logits = self.out_mode(q.squeeze(1))  # Mode logits (shared) 
        exp_logits = []  # Expert logits per mode
        for m in range(2):
            token_m = self.expert_token[m].unsqueeze(0).expand(N, -1, -1)  
            exp_logits.append(self._attend_expert_tokens(q, token_m))      
        exp_logits = torch.stack(exp_logits, dim=1) 
        return mode_logits, exp_logits


# ---------------------------------------------------------------------------
# 3) MoE module with Graph-adaptive gating network
# ---------------------------------------------------------------------------
class HybridMoE(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        dag_dim: int,
        ready_task_dim: int,
        sla_gamma_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 128,
        mode_top_k: int = 2,             
        gating_d_model: int = 128,
        gating_heads: int = 2,

        # New knobs
        mode_source: str = "gating",    
        shared_expert: bool = True,     
        gating_nn_type: str = "attn",  
        post_train_exp: bool = True,   
        model_paths=None,
        model_name: str = "GATES",      
    ) -> None:
        super().__init__()

        if mode_source not in {"parallel"}: 
            raise ValueError("mode_source must be 'parallel' ") 
        if model_name not in {"GATES"}:
            raise ValueError("model_name must be GATES")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if num_experts < top_k:
            raise ValueError("num_experts must be >= top_k")

        self.num_experts = num_experts
        self.top_k = top_k
        self.mode_top_k = max(1, mode_top_k)
        self.mode_source = mode_source
        self.shared_expert = shared_expert
        self.gating_nn_type = gating_nn_type
        self.post_train_exp = post_train_exp
        self.model_paths = model_paths
        self.model_name = model_name
        self._alt_next = 0

        # Graph-adaptive gating network
        self.gating = HybridRoutingGating(
            action_dim,
            dag_dim,
            ready_task_dim,
            sla_gamma_dim,
            d_model=gating_d_model,
            num_experts=num_experts,
            num_heads=gating_heads,
        )

        # Experts pool 
        hidden = expert_hidden_dim
        if self.model_paths is None:
            self.experts_parallel = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(action_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1)
                )
                for _ in range(num_experts)
            ])
        else:
            # 1)Load pre-trained weights for initializing parallel MLP experts:
            if self.model_name == "GATES":
                from policy.wf_model_02 import SelfAttentionEncoder  # the model of GATES
                mlp_expert_model = SelfAttentionEncoder(task_fea_size=4, vm_fea_size=4, output_size=1, d_model=16,
                                                        att_heads=2, att_en_layers=2, d_ff=128, gat_heads=2, dropout=0.1)

            self.experts_parallel = nn.ModuleList()
            for i in range(num_experts):
                model_path = model_paths[i % len(model_paths)]
                ckpt = torch.load(model_path, map_location="cpu")  # all state_dict of GATES
                state_dict_full = ckpt.get('state_dict', ckpt)
                mlp_expert_model_weights = {k.split("model.", 1)[1]: v for k, v in state_dict_full.items()
                                            if k.startswith("model.")}  # the state_dict of the model of GATES
                mlp_expert_model.load_state_dict(mlp_expert_model_weights, strict=True)
                exp = copy.deepcopy(mlp_expert_model.priority)  # only use the PMM as the MLP expert
                if not self.post_train_exp:
                    for p in exp.parameters():
                        p.requires_grad = False  
                    exp.eval() 
                self.experts_parallel.append(exp)

    @torch.no_grad()
    def _pick_mode_forced(self, N: int, device: torch.device) -> torch.Tensor: 
        mode_w = torch.zeros(N, 2, device=device)
        if self.mode_source == "parallel":
            mode_w[:, 0] = 1
        return mode_w

    def _run_parallel_cluster(
        self,
        actions: torch.Tensor,      # [n, D]
        idx: torch.Tensor,          # [n] index of selected expert for each action (0 … E‑1)
        weights: torch.Tensor,      # [n, 1] final per‑action weight (mode × expert softmax)
    ) -> torch.Tensor:
        out = torch.zeros(actions.size(0), 1, device=actions.device)
        for e, expert in enumerate(self.experts_parallel):
            mask = (idx == e)
            if mask.any():
                out[mask] += expert(actions[mask]) * weights[mask]
        return out

    # forward 
    def forward(
        self,
        actions: torch.Tensor,   
        dag: torch.Tensor,       
        ready: torch.Tensor,    
        sla_gamma: torch.Tensor  
    ) -> torch.Tensor:
        N = actions.size(0)
        device = actions.device
        out = torch.zeros(N, 1, device=device)

        # 1) Compute gating logits
        if self.gating_nn_type == "attn":
            if self.shared_expert:
                a_ctx = actions.mean(dim=0, keepdim=True)  
                mode_logits_1, exp_logits_1 = self.gating(a_ctx, dag, ready, sla_gamma)  
                mode_logits = mode_logits_1.expand(N, -1)                   
                exp_logits  = exp_logits_1.expand(N, -1, -1)             
            else:
                mode_logits, exp_logits = self.gating(actions, dag, ready, sla_gamma)    
        else:
            raise ValueError("gating_nn_type should be 'attn'")

        # 2) mode weights
        mode_w = self._pick_mode_forced(N=N, device=device)  

        # 3) mixture 
        mask_p = mode_w[:, 0] > 0
        if mask_p.any():
            act_p, mw_p = actions[mask_p], mode_w[mask_p, 0:1]       
            logits_p = exp_logits[mask_p, 0]                          
            top_val, top_idx = torch.topk(logits_p, k=self.top_k, dim=-1)  
            exp_w = F.softmax(top_val, dim=-1) * mw_p                  
            out_p = torch.zeros(act_p.size(0), 1, device=act_p.device)
            for kk in range(self.top_k):
                out_p += self._run_parallel_cluster(act_p, top_idx[:, kk], exp_w[:, kk:kk+1])
            out[mask_p] += out_p

        return out

