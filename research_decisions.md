# Research Decisions Log
**Project:** Embedding-Space vs. Accuracy-Based Drift Detection for Open-Source LLMs  
**Student:** MRA14 | **Started:** November 2026

---

## Decision 1 — Topic Continuity
| Field | Detail |
|---|---|
| **Decision** | Continue in the LLM Performance Drift research area from Lab 01 review |
| **Reason** | Assignment requires continuity; review documented clear gap in monitoring methodology |
| **Source / Evidence** | VERIFIED — Chen et al. (2023) arXiv:2307.09009 Section 5 explicitly calls for cost-efficient monitoring approaches |
| **Next action** | Extract top 3 candidate gaps from review using Prompt 02 |

## Decision 2 — Study Pathway Selection
| Field | Detail |
|---|---|
| **Decision** | Pathway B — Controlled Comparative Study |
| **Reason** | Pathway A (reproduction+extension) requires replicating Chen et al. API calls to GPT-4 — no API access; Pathway C (modification) requires a specific baseline to modify; Pathway B is feasible with open-source models |
| **Source / Evidence** | VERIFIED — GPT-2-medium and DistilGPT-2 are available on Hugging Face under MIT / Apache 2.0 |
| **Next action** | Freeze experiment protocol before viewing any test results |

## Decision 3 — Dataset Choice: TriviaQA
| Field | Detail |
|---|---|
| **Decision** | Use TriviaQA `rc.nocontext` split |
| **Reason** | (1) Freely available, Apache 2.0, no signup required. (2) Multiple answer aliases per question enable robust EM/F1 scoring. (3) Factual QA is directly sensitive to knowledge drift. |
| **Source / Evidence** | VERIFIED — `datasets.load_dataset("trivia_qa", "rc.nocontext")` loads without credentials |
| **Alternatives rejected** | MMLU: requires careful license check; NaturalQuestions: larger download; SQuAD: single-document context available, removes "no-context" drift effect |
| **Next action** | Implement keyword classifier for drift partitioning |

## Decision 4 — Drift Simulation via Keyword Partitioning
| Field | Detail |
|---|---|
| **Decision** | Classify TriviaQA questions as "stable" (geography, science, history keywords) or "drift" (sports, entertainment, person-centric keywords) using a keyword heuristic frozen before result inspection |
| **Reason** | True temporal drift (re-running same API over months) is not feasible in a course project. Domain shift is the closest tractable proxy and is explicitly acknowledged as a limitation. |
| **Source / Evidence** | INFERRED — domain shift causing performance drop is consistent with Chen et al. (2023) observation that temporally variable question types drift more |
| **Next action** | Implement `classify_question()` in `src/data_loader.py`; freeze keyword lists in config |

## Decision 5 — Embedding Model: all-MiniLM-L6-v2
| Field | Detail |
|---|---|
| **Decision** | Use `sentence-transformers/all-MiniLM-L6-v2` for MMD-MON embeddings |
| **Reason** | (1) Small (22M params, 384-dim embeddings). (2) State-of-the-art on SBERT benchmarks for semantic similarity. (3) License: Apache 2.0. (4) Used as the input prompt encoder, so no dependency on monitored model's internals. |
| **Source / Evidence** | VERIFIED — available at https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| **Next action** | Add to requirements.txt |

## Decision 6 — Threshold Calibration at 95th Percentile
| Field | Detail |
|---|---|
| **Decision** | Calibrate all detector thresholds at the 95th percentile of the respective statistic over non-overlapping reference windows |
| **Reason** | Ensures nominal FPR ≤ 5% under the null hypothesis of no drift; consistent across all three monitors; calibrated once before test-stream inspection |
| **Source / Evidence** | INFERRED from standard practice in concept drift literature (Gama et al., 2014) |
| **Next action** | Implement `calibrate_threshold()` in `src/drift_detector.py` |

## Decision 7 — Evaluation Metrics
| Field | Detail |
|---|---|
| **Decision** | Primary drift metric: AUC-ROC. Secondary: Average Detection Delay (ADD), FPR@TPR=0.90. Task metric: Exact Match (EM), token-level F1 |
| **Reason** | AUC-ROC is threshold-independent; ADD measures operational relevance; EM/F1 follows SQuAD convention (Rajpurkar et al., 2016) which TriviaQA also uses |
| **Source / Evidence** | VERIFIED — TriviaQA evaluation script uses EM and F1; Joshi et al. (2017) confirm this |
| **Next action** | Implement in `src/evaluator.py` |

## Decision 8 — Protocol Freeze (Pre-registration)
| Field | Detail |
|---|---|
| **Decision** | Primary metric = AUC-ROC on GPT-2-medium. Test split and partition keyword lists frozen before any model runs on test stream. No hyperparameter changes after test-stream inspection. |
| **Reason** | Prevents test-set tuning; assignment explicitly requires pre-registration |
| **Source / Evidence** | Assignment instructions Section 7 (Prompt 07); standard experimental practice |
| **Next action** | Commit `configs/experiment_config.json` before running `03_run_experiment.py` |
