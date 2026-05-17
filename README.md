# Patient–Trial Matching for Biomarker-Stratified Oncology

> Where current LLM matching approaches break — and how to measure it honestly.

## Why this project

- Fewer than 5% of adult cancer patients participate in clinical trials.
- 56% of patients who don't enroll cite *no suitable trial available at their treatment location*.
- More than half of current oncology trials use biomarkers for patient selection, and biomarker-driven trials are most common in rare tumor types, lung, and breast cancer.

Most patient–trial matching demos report aggregate accuracy that hides the cases that matter most in production: multi-criterion biomarker logic, negations, prior-therapy line counting, and rare-disease prevalence. This project builds a matcher *and* the evaluation framework that exposes where each approach fails — stratified by the failure modes that oncology recruitment teams actually care about.

## What's inside

Three matching approaches, evaluated against a hand-curated gold standard:

1. **Keyword/regex baseline** — the strawman, useful for showing the floor.
2. **Semantic retrieval** — sentence-transformer embeddings + ChromaDB.
3. **LLM reranker** — retrieval + structured-output LLM judge, tested with both Anthropic Claude (paid) and Google Gemini Flash (free) so the cost–quality frontier is honest.

Plus a **synthetic oncology patient generator** that extends Synthea with realistic molecular biomarker profiles (EGFR / KRAS / ALK / HER2 / MSI / PD-L1 status) sampled from published prevalence rates by tumor type. Released as a standalone module.

## Headline results

> Filled in at Week 5. Hero chart goes here.

| Method                | Overall P@10 | Biomarker-simple P@10 | Multi-criterion P@10 | Cost / query |
|-----------------------|--------------|-----------------------|----------------------|--------------|
| Keyword               | TBD          | TBD                   | TBD                  | $0           |
| Semantic retrieval    | TBD          | TBD                   | TBD                  | $0           |
| Retrieval + Gemini    | TBD          | TBD                   | TBD                  | TBD          |
| Retrieval + Claude    | TBD          | TBD                   | TBD                  | TBD          |

## Failure-mode taxonomy

Hand-tagged on every gold-standard pair. Frequencies and per-method error rates reported in `results/failure_analysis.md`.

- **Biomarker logic** — combined / negated / quantitative thresholds (e.g., "PD-L1 CPS ≥10 AND no EGFR sensitizing mutations")
- **Prior therapy lines** — counting and class constraints ("≥2 prior platinum-based regimens, no prior immunotherapy")
- **Stage + histology subtype** — multi-attribute constraints ("Stage IIIB/IV non-squamous NSCLC")
- **Performance status** — ECOG/Karnofsky lookups
- **Temporal constraints** — recency ("within last 6 months")
- **Geographic feasibility** — site distance, not just eligibility
- **Contradictory or ambiguous criteria** — what the model does when criteria themselves are unclear

## Project structure

```
patient-trial-matching/
├── README.md
├── pyproject.toml         # uv-managed dependencies
├── .env.example
├── .gitignore
├── data/
│   ├── raw/               # gitignored — API pulls, Synthea output
│   ├── interim/           # gitignored — parsed parquet
│   └── gold/              # COMMITTED — hand-annotations (the project's moat)
├── notebooks/
│   ├── 01_data_pull.ipynb
│   ├── 02_patient_generation.ipynb
│   ├── 03_baseline_matcher.ipynb
│   ├── 04_embedding_retrieval.ipynb
│   ├── 05_llm_reranker.ipynb
│   └── 06_evaluation.ipynb
├── src/
│   ├── data/              # API clients, parsers, biomarker augmentation
│   ├── matchers/          # baseline, embedding, llm
│   ├── eval/              # metrics, failure-mode tagging
│   └── viz/               # plotting helpers
├── tests/
└── results/
    ├── figures/
    └── tables/
```

## Quickstart

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <your-repo-url>
cd patient-trial-matching
uv sync

# Set up API keys (optional for Week 1 data pulls)
cp .env.example .env
# edit .env with your ANTHROPIC_API_KEY and/or GOOGLE_API_KEY

# Pull trials and inspect
uv run jupyter lab notebooks/01_data_pull.ipynb
```

## Status

| Week | Milestone                          | Status |
|------|------------------------------------|--------|
| 1    | Data pulled, biomarker augmenter   | 🟡 in progress |
| 2    | Embedding retrieval + gold start   | ⚪ not started |
| 3    | LLM reranker + failure tagging     | ⚪ not started |
| 4    | Stratified evaluation              | ⚪ not started |
| 5    | Writeup & launch                   | ⚪ not started |

## Disclaimers

- **Synthetic patients only.** No real patient data is used. Synthea patients with augmented biomarker profiles are statistically plausible, not clinically validated.
- **Not for clinical use.** This is a methods exploration. Any deployment in a clinical recruitment context requires regulatory, clinical, and ethical review.
- **Independent project.** Not affiliated with my employer; built on public data and personal time.

## License

MIT for code. Hand-annotated gold-standard labels released under CC BY 4.0 — please credit if you use them.
