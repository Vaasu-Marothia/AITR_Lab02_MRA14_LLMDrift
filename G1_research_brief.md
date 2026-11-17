# Gate G1 — Research Brief
**Student:** MRA14 | **Assignment:** AITR Lab 02 | **Date:** November 2026

---

| Field | Detail |
|---|---|
| **Review topic & key papers** | LLM Performance Drift — temporal and distributional degradation in large language models. Key references verified from review: (1) Chen et al. (2023) *How Is ChatGPT's Behavior Changing over Time?* arXiv:2307.09009; (2) Liang et al. (2023) *HELM* TMLR; (3) Gama et al. (2014) *A Survey on Concept Drift Adaptation* ACM CSUR; (4) Lu et al. (2018) *Learning under Concept Drift* IEEE TKDE; (5) Ouyang et al. (2022) *InstructGPT* NeurIPS. |
| **Research gap** | Chen et al. (2023) [VERIFIED from original paper] document significant behavioral drift in proprietary APIs (GPT-3.5/4) but rely on expensive repeated labelled evaluation. It is **unknown** whether lightweight, label-free distributional metrics can serve as early-warning drift indicators in open-source LLMs — a gap noted implicitly in Section 5 of the HELM paper (Liang et al., 2023) under "monitoring and evaluation costs". |
| **Research question** | *Do embedding-space distributional metrics — Jensen-Shannon Divergence over output logit distributions (JSD-MON) and Maximum Mean Discrepancy over sentence embeddings (MMD-MON) — provide earlier and more reliable detection of task-performance degradation than accuracy-based monitoring (ACC-MON) in open-source LLMs under controlled domain distribution shift?* |
| **Supporting sub-question** | Does increasing the sliding window size improve detection reliability at the cost of detection delay? (Ablation study) |
| **Study pathway** | **B — Controlled comparative study**: three monitoring methods evaluated under a consistent, frozen protocol on two open-source model scales. |
| **Dataset / measurement setup** | TriviaQA `rc.nocontext` split (Joshi et al., 2017) — Apache 2.0 license, freely available via Hugging Face `datasets`. Partitioned into reference window (n=500), stable stream (n=350), and drift stream (n=350) using a pre-specified keyword heuristic frozen before result inspection. |
| **Baseline / comparators** | (1) **ACC-MON** — rolling accuracy monitoring [primary baseline, requires labels]. (2) **JSD-MON** — Jensen-Shannon Divergence over output probability distributions [label-free]. (3) **MMD-MON** — Maximum Mean Discrepancy over Sentence-BERT embeddings [label-free]. All compared on the same test stream. |
| **Primary metric & control** | Primary: AUC-ROC of drift detector (binary in-dist / drifted label). Secondary: Average Detection Delay, FPR@TPR=0.90. Task performance: Exact Match (EM), token-level F1. Held constant: dataset partition, random seed (42), generation parameters, threshold calibration procedure. |
| **Computing & risks** | Hardware: single GPU ≥4 GB VRAM (or CPU). Models: `gpt2-medium` (345M), `distilgpt2` (82M) — freely available, no license restriction. Estimated run time: 45 min (GPU) / 3 hr (CPU). **Risk:** TriviaQA download fails → use cached Hugging Face dataset. **Risk:** GPU OOM → reduce batch size to 1 in config. |
| **First executable milestone** | Environment test (`scripts/01_environment_test.py`) passes; TriviaQA sample of 5 questions loads; GPT-2-medium generates a response to "Q: What is the capital of France?\nA:". |

---

**Status:** READY TO PROCEED — question is narrow, literature-supported, experimentally testable, and feasible on consumer hardware.
