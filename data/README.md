# Dataset Documentation

## Primary Dataset: TriviaQA (rc.nocontext)

| Field | Value |
|---|---|
| **Source** | Joshi et al. (2017), ACL 2017 |
| **HuggingFace ID** | `trivia_qa`, config `rc.nocontext` |
| **Split used** | `validation` (n=1500 sampled, seed=42) |
| **License** | Apache 2.0 |
| **Download** | Automatic via `datasets.load_dataset("trivia_qa", "rc.nocontext")` |
| **Requires signup** | No |
| **Contains PII** | No |
| **Ethical review** | Not required — publicly available, non-sensitive benchmark |

## Partitioning Protocol (Frozen)

Keyword classifier (frozen before test-stream results inspected):

**Stable questions** (in-distribution): contain keywords from —
`["capital of", "largest", "highest", "deepest", "longest", "smallest",
 "made of", "chemical", "symbol for", "founded in", "born in", "century",
 "named after", "atomic", "speed of", "distance"]`

**Drift questions** (drifted): contain keywords from —
`["who won", "who sang", "who directed", "who played", "who scored",
 "which team won", "best known for", "whose", "who was awarded",
 "who became", "who is the", "who was the first to"]`

**Neutral**: all remaining questions, excluded from drift stream, used to pad reference window.

## Partition Sizes

| Split | n | Label |
|---|---|---|
| Reference window | 500 | in-distribution |
| Stable stream | 350 | in-distribution (is_drift=0) |
| Drift stream | 350 | drifted (is_drift=1) |

## Important Notes

- **Raw data files are NOT committed to this repository** to respect Hugging Face terms.
- Generated CSV partitions (under `data/partitions/`) are derived from the public dataset and are safe to commit.
- Random seed 42 is used throughout; changing it will produce different partitions.
- If TriviaQA download fails (network issue), cached version at `~/.cache/huggingface/datasets/trivia_qa/` is used automatically.
