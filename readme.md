# Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts

This repository provides the **DEFT-specific policy modules** for our ICLR 2026 method for dynamic cloud workflow scheduling under varying deadlines. Use the original [GATES](https://github.com/yashenCS/GATES) repository as the base project for training, evaluation, and reproduction, and integrate the modules released here into that framework.

The official website of this paper: [DEFT_ICLR_2026](https://iclr.cc/virtual/2026/poster/10006550)

[![DEFT Poster](ICLR_2026_poster_ya.png)](ICLR_2026_poster_ya.pdf)

---

## Overview

Dynamic cloud workflow scheduling requires assigning ready tasks from dynamically arriving DAG-structured workflows to heterogeneous virtual machines (VMs) under changing system conditions and workflow deadlines. The goal is to minimize the **total scheduling cost**, including both **VM rental cost** and **SLA penalty**.

DEFT introduces a **deadline-aware Mixture-of-Experts (MoE) policy**. Instead of using a single fixed priority mapping pathway, DEFT maintains multiple experts specialized for different deadline tightness regimes and uses a **graph-adaptive gating network** to select the most suitable expert at each scheduling step.

The main ideas of DEFT are:

- **Deadline-aware expert specialization.** Multiple experts are pre-trained under different deadline settings so that each expert learns a distinct scheduling preference.
- **Graph-adaptive gating.** A dedicated gating network uses workflow DAG information, ready-task features, VM state embeddings, and deadline urgency to route decisions to the most suitable expert.
- **Two-phase training.** Experts are first pre-trained independently, then integrated into the full DEFT policy and jointly optimized together with the gating network and state embedding module.

---

## What Is Included in This Release

The three Python files are located at the root of this repository:

- [`moe.py`](moe.py): DAG encoder, graph-adaptive gating network, and MoE routing module.
- [`stateEmbeddingLearning.py`](stateEmbeddingLearning.py): GATES-style state embedding module and intermediate representations for expert routing.
- [`wf_model.py`](wf_model.py): DEFT policy (`WFPolicy`), combining state embeddings, the DAG encoder, MoE routing, and action selection.

Use GATES for the simulator, environment, training/evaluation scripts, configuration, and ES optimizer.

---

## How to Use DEFT

Here, **gamma controls workflow deadline tightness**: smaller values give tighter deadlines, while larger values give looser deadlines.

1. **Stage 1 — expert pre-training with fixed gamma.** Train GATES policies independently, using a different fixed gamma for each expert to cover different deadline regimes. Use the resulting checkpoints to initialize the DEFT experts through `model_paths` in `wf_model.py`.

2. **Stage 2 — joint training with varying gamma.** Load the pre-trained experts and jointly train them with the state embedding module, DAG encoder, and gating network. **Design and generate the dynamic-gamma instances**: gamma should vary across instances, while remaining fixed for each workflow.

---

## Citation

If you find this project useful for your research, please consider giving a star and citing the following paper in your future works:

```bibtex
@inproceedings{
shen2026deft,
title={Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts},
author={Ya Shen and Gang Chen and Hui Ma and Mengjie Zhang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=yVFOdLjd7V}
}
```

---

## Acknowledgement

DEFT is implemented on top of the [GATES](https://github.com/yashenCS/GATES) backbone. We thank the contributors from the [GATES](https://github.com/yashenCS/GATES) codebase for providing the simulator, environment, and the training framework.
