# Submission Checklist — AITR Lab 02 (MRA14)

Suggested submission folder name: `AITR_Lab02_MRA14_LLMDrift`

---

## Item A — Research Paper

- [ ] `paper/main.tex` compiles without errors
- [ ] Author name / enrollment number filled in
- [ ] Institution name filled in
- [ ] Tables 1–3 updated with **actual experimental results** (not illustrative values)
- [ ] `paper/figures/` contains all 4 figures from `04_evaluate_results.py`
- [ ] Abstract matches final results
- [ ] All `[TO VERIFY]` markers resolved
- [ ] `paper/main.pdf` generated and readable

---

## Item B — GitHub Repository

- [ ] Repository is public (or shared with instructor)
- [ ] `README.md` present with correct run commands
- [ ] `.gitignore` excludes model weights and raw data
- [ ] `requirements.txt` lists all dependencies with versions
- [ ] No API keys, secrets, or restricted data committed
- [ ] At least one commit before the submission deadline

---

## Item C — Experimental Evidence

- [ ] `results/baseline_gpt2medium.jsonl` present
- [ ] `results/baseline_distilgpt2.jsonl` present
- [ ] `results/stream_gpt2medium_w150.jsonl` present
- [ ] `results/stream_distilgpt2_w150.jsonl` present
- [ ] `results/table1_task_performance.csv` present
- [ ] `results/table2_drift_detection.csv` present
- [ ] `results/table3_ablation.csv` present (run `--ablation` first)
- [ ] `configs/experiment_config.json` matches what was actually run

---

## Item D — Research Provenance

- [ ] `G1_research_brief.md` — research brief submitted (Gate G1)
- [ ] `baseline_validation.md` — baseline validation checked (Gate G2)
- [ ] `research_decisions.md` — 8 decisions documented with evidence labels
- [ ] `ai_assistance_log.md` — at least 3 real AI-interaction records + 3 AI limitations

---

## Item E — Demonstration (5 minutes)

Suggested walkthrough order:
1. Show `G1_research_brief.md` — research question and gap (30 sec)
2. Open `notebooks/00_exploration.ipynb` — run mini experiment live (2 min)
3. Show `results/figures/roc_gpt2medium.png` — MMD-MON AUC-ROC (30 sec)
4. Show `results/figures/ablation_gpt2medium.png` — window ablation (30 sec)
5. Open `paper/main.pdf` — show Tables 1–3 and discuss one result (1 min)
6. Answer one question from `ai_assistance_log.md` — which AI suggestion was rejected and why (30 sec)

---

## Student Final Self-Check (from assignment)

- [ ] Research question follows directly from earlier review, with original source evidence
- [ ] Claims of gap and novelty are verified or carefully qualified
- [ ] Dataset, license and evaluation protocol are recorded
- [ ] Baseline runs, has been checked for correctness and is preserved for comparison
- [ ] Experimental design answers the stated research question
- [ ] Every figure, metric and result is traceable to real code output
- [ ] At least one deeper experiment (window ablation) reported with limitations
- [ ] LaTeX manuscript distinguishes prior work, my changes, and experimental findings
- [ ] Repository contains genuine commands, dependencies, configurations, no exposed secrets
- [ ] AI prompt log and technical decisions are authentic; work can be explained and demonstrated
