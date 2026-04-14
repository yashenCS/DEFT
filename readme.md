# Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts

This repository provides the implementation for DEFT, our ICLR 2026 method for dynamic cloud workflow scheduling under varying deadlines. DEFT should be used together with the original [GATES](https://github.com/yashenCS/GATES) codebase, which already contains the simulator, environment, training pipeline, and evaluation scripts. 

The official website of this paper at: [DEFT_ICLR_2026](https://iclr.cc/virtual/2026/poster/10006550)


[![DEFT Poster](ICLR_2026_poster_ya.png)](ICLR_2026_poster_ya.png)

---

## Overview

Dynamic cloud workflow scheduling requires assigning ready tasks from dynamically arriving DAG-structured workflows to heterogeneous virtual machines (VMs) under changing system conditions and workflow deadlines. The goal is to minimize the **total scheduling cost**, including both **VM rental cost** and **SLA penalty**.

DEFT introduces a **deadline-aware Mixture-of-Experts (MoE) policy**. Instead of using a single fixed priority mapping pathway, DEFT maintains multiple experts specialized for different deadline tightness regimes and uses a **graph-adaptive gating network** to select the most suitable expert at each scheduling step.

The main ideas of DEFT are:

- **Deadline-aware expert specialization.** Multiple experts are pre-trained under different deadline settings so that each expert learns a distinct scheduling preference.
- **Graph-adaptive gating.** A dedicated gating network uses workflow DAG information, ready-task features, VM state embeddings, and deadline urgency to route decisions to the most suitable expert.
- **Two-phase training.** Experts are first pre-trained independently, then integrated into the full DEFT policy and jointly optimized together with the gating network and state embedding module.

DEFT **fundamentally redesigns** the decision-making policy network by introducing a **Mixture-of-Experts architecture** and a **graph-adaptive gating network** for context-dependent expert routing, and an effecttive **two-phase training procedure** for poicy learning. 

---

## What Is Included in This Release

This release only contains the DEFT-specific module needed to reproduce the policy-side contribution of the paper. Included files:

- `MoE/moe.py`  
  Implements the main MoE components used by DEFT, including:
  - the DAG encoder,
  - the graph-adaptive gating network, and
  - the hybrid MoE routing module.

- `MoE/stateEmbeddingLearning.py`  
  Contains the state embedding module (SEM) used by DEFT. This module follows the GATES backbone design and exposes the intermediate representations required by the DEFT gating and expert-routing logic.

- `MoE/wf_model.py`  
  Provides the DEFT policy network (`WFPolicy`) that combines:
  - the GATES-style state embedding module,
  - the DAG encoder,
  - the MoE module, and
  - the action selection logic for workflow scheduling.

This release does not duplicate the following parts, which should be obtained from [GATES](https://github.com/yashenCS/GATES):

  - simulator and environment implementation,
  - training and evaluation scripts,
  - configuration system,
  - optimizer / ES training framework,
  - policy base classes and project scaffolding.

---

## How to Use DEFT 

To run DEFT, please start from the original [GATES repository](https://github.com/yashenCS/GATES) and then integrate the DEFT-specific modules from this release.

A high-level integration workflow is:

1. **Clone and set up GATES**
   - Use the GATES repository as the main project root.
   - Follow its environment setup and dependency instructions.

2. **Add the DEFT modules to the GATES project**
   - Copy the `MoE/` directory from this release into the GATES project.
   - Keep the original GATES project structure intact.

3. **Connect the DEFT policy to the GATES training framework**
   - The DEFT implementation in `MoE/wf_model.py` defines a `WFPolicy` compatible with the existing workflow scheduling setting.

4. **Reuse the GATES simulator and training scripts**
   - Continue to use the original simulator, environment, config system, and ES-based training backbone from GATES.
   - DEFT is intended to run inside that framework rather than as an independent training codebase. 

5. **Prepare expert initialization checkpoints**
   - `MoE/wf_model.py` expects multiple pre-trained model paths for expert initialization.
   - These experts are initialized from pre-trained GATES-style policy checkpoints trained under different deadline settings.
   - You should replace the placeholder paths in the code with your own checkpoint paths.

6. **Run DEFT training inside the GATES framework**
   - Stage 1: pre-train experts under different deadline regimes.
   - Stage 2: load these experts into DEFT and jointly train the state embedding module, gating network, and experts using the existing GATES training backbone.

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
