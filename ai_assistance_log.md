# AI Assistance Log (Prompt 18 — Human–AI Accountability)
**Project:** Embedding-Space vs. Accuracy-Based Drift Detection for Open-Source LLMs  
**Student:** MRA14 | **Assignment:** AITR Lab 02

---

## Overview
AI tools (Claude/Gemini) were used as a supervised research assistant for ideation, code scaffolding, debugging, and manuscript drafting. All AI suggestions were verified, modified, or rejected by the student. Final claims, results, and experimental decisions are the student's responsibility.

---

## Activity Log

### Activity 1 — Gap Identification (Prompt 02)
| Field | Detail |
|---|---|
| **Task** | Extract evidence-based research gaps from literature review |
| **Tool** | Antigravity (Claude/Gemini) |
| **AI Proposal** | Suggested 5 gaps including: (a) label-free monitoring metrics, (b) open-source drift measurement, (c) robustness under paraphrase attacks, (d) multi-task drift, (e) drift in embedding-only pipelines |
| **Human Decision** | Selected gap (a)+(b): lightweight label-free drift detection for open-source LLMs. Gaps (c)–(e) flagged as outside scope or requiring different baselines |
| **Verification** | Confirmed gap is genuine by checking Chen et al. (2023) Section 5 discussion — they note monitoring costs without proposing a solution |
| **Status** | ✅ VERIFIED |

### Activity 2 — Drift Detection Algorithm Design (Prompt 09)
| Field | Detail |
|---|---|
| **Task** | Design the three monitoring classes (ACC-MON, JSD-MON, MMD-MON) |
| **Tool** | Antigravity |
| **AI Proposal** | Generated initial skeleton for `AccuracyMonitor` and `JSDMonitor`; suggested using sliding deque |
| **Human Decision** | Accepted sliding deque for efficiency. Modified `JSDMonitor` to average distributions over the window rather than using the last sample only — the AI's first version used only the current sample, which was too noisy |
| **Verification** | Unit-tested JSD values using known distributions (uniform vs peaked); confirmed bounded in [0, ln2] |
| **Status** | ✅ VERIFIED — human correction applied |

### Activity 3 — MMD Implementation
| Field | Detail |
|---|---|
| **Task** | Implement unbiased MMD² estimator |
| **Tool** | Antigravity |
| **AI Proposal** | Generated MMD code using biased estimator (did not exclude diagonal in within-set sums) |
| **Human Decision** | Corrected to unbiased estimator by adding `np.fill_diagonal(kxx, 0.0)` and adjusting normalization to `m*(m-1)`. Verified against Gretton et al. (2012) Equation 5 in the original paper |
| **Verification** | Tested: MMD²(X, X) ≈ 0 for same distributions; MMD²(normal, uniform) > 0 ✓ |
| **Status** | ✅ VERIFIED — human correction applied; original paper consulted |

### Activity 4 — Threshold Calibration Strategy
| Field | Detail |
|---|---|
| **Task** | Decide how to calibrate detection thresholds |
| **Tool** | Antigravity |
| **AI Proposal** | Suggested fixed thresholds (e.g. JSD > 0.05) based on "typical values" |
| **Human Decision** | Rejected fixed thresholds — they are dataset/model-specific. Instead implemented calibration at the 95th percentile of reference-window statistics, following Gama et al. (2014) recommendation for adaptive thresholds |
| **Verification** | Confirmed the calibration procedure yields nominally 5% FPR on held-out reference sub-windows |
| **Status** | ✅ VERIFIED — AI suggestion overridden; replaced with principled approach from literature |

### Activity 5 — LaTeX Paper Drafting (Prompt 15)
| Field | Detail |
|---|---|
| **Task** | Draft IEEE-format paper sections |
| **Tool** | Antigravity |
| **AI Proposal** | Drafted Introduction, Related Work, and Methodology sections |
| **Human Decision** | Revised Related Work to remove two references that were INFERRED (not in original review): deleted one citation to a paper I could not access, flagged it as [TO VERIFY]. Methodology equations were checked line-by-line against Lin (1991) and Gretton et al. (2012) |
| **Verification** | All remaining references are from original papers verified during the literature review |
| **Status** | ✅ VERIFIED — AI draft substantially revised |

### Activity 6 — Debugging: Empty Drift Pool
| Field | Detail |
|---|---|
| **Task** | Fix `ValueError: Not enough drift questions` in `data_loader.py` |
| **Tool** | Antigravity |
| **AI Proposal** | Suggested expanding DRIFT_KEYWORDS list; also suggested lowering `n_drift_stream` |
| **Human Decision** | Chose to expand DRIFT_KEYWORDS thoughtfully — added "who became" and "who is the" which are semantically valid drift indicators. Did NOT lower n_drift_stream (would change the pre-registered protocol). Logged in research_decisions.md |
| **Verification** | Reran with new keywords; drift pool now has >400 questions; classification spot-checked on 20 examples |
| **Status** | ✅ VERIFIED — both fix options evaluated; principled choice made |

---

## AI Limitations Encountered

1. **Fabricated numbers**: When asked "what accuracy should GPT-2 achieve on TriviaQA?", AI gave "about 35–40%" — this was incorrect. I verified against literature and experimental baseline: actual values are 10–15% EM for no-context autoregressive LMs. The AI was confidently wrong.

2. **Biased estimator**: AI generated a biased MMD² estimator (did not exclude diagonal). Required manual verification against the original Gretton et al. paper to identify and fix.

3. **Overstated novelty**: AI described the study as "the first work to apply MMD to LLM drift detection." I could not verify this claim — it was flagged as UNKNOWN and removed from the paper. The claim was replaced with "extending MMD-based drift detection to LLM monitoring."

---

## Human Reasoning & Decisions

- **Choice of drift proxy**: The keyword-based domain partition as a drift proxy is my own decision; AI suggested "paraphrase attacks" which would not simulate realistic deployment drift.
- **Window size to test**: I chose the specific window sizes {50, 100, 150, 200, 250} for the ablation based on the reference window size (500); testing larger windows than the reference would be statistically unsound.
- **Paper framing**: The framing as "label-free early warning indicator" rather than "unsupervised drift detector" is my contribution to the narrative — the distinction is important and was not suggested by AI.
