"""
Synthea oncology biomarker augmenter.

Synthea generates plausible patient histories but does NOT include molecular
biomarker status for cancer patients. This module samples biomarkers from
published prevalence rates so generated oncology patients have:

    - mutation status (EGFR, KRAS, ALK, ROS1, BRAF, HER2, BRCA1/2, ...)
    - expression status (PD-L1 TPS, HER2 IHC, ER/PR)
    - molecular phenotypes (MSI, TMB)
    - ECOG performance status
    - prior treatment lines

These augmented patients are *not* clinically validated. They are
statistically plausible synthetic profiles for use in matching evaluation.

Prevalence rates are sourced from peer-reviewed literature and standard
oncology references. Citations live in `references/biomarker_prevalence.md`.

Key references used (caveat: rough population-level estimates):
- NSCLC: Chevallier et al. 2021 (WJCC); LCMC consortium
- Breast: ACS Cancer Facts & Figures 2024
- CRC: Phipps et al. 2015; AACR Project GENIE
- Gastric: Bang et al. (ToGA trial); CheckMate-649

THIS IS A STARTER. Refine prevalence rates with current literature before
publishing — list it as a known limitation if you don't.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

CancerType = Literal[
    "nsclc",         # non-small cell lung cancer
    "sclc",          # small cell lung cancer
    "breast",
    "colorectal",
    "gastric",
    "pancreatic",
    "ovarian",
    "cholangiocarcinoma",
    "mesothelioma",
    "sarcoma",
]


# ---------------------------------------------------------------------------
# Biomarker prevalence configuration
# ---------------------------------------------------------------------------
#
# Each entry: prevalence is the marginal probability among that cancer type.
# Independence is assumed for simplicity — a real implementation should model
# correlation structure (e.g. EGFR and KRAS are largely mutually exclusive in
# NSCLC). Document this limitation in your writeup.

NSCLC_BIOMARKERS = {
    "egfr_mutation": 0.15,           # sensitizing mutations (exon 19 del / L858R)
    "kras_g12c": 0.13,
    "kras_other": 0.17,              # other KRAS mutations
    "alk_rearrangement": 0.05,
    "ros1_rearrangement": 0.02,
    "braf_v600e": 0.02,
    "her2_mutation": 0.03,
    "met_exon14": 0.03,
    "ret_fusion": 0.02,
    "ntrk_fusion": 0.005,
}

NSCLC_EXPRESSION = {
    # PD-L1 TPS distribution (rough): <1% ~40%, 1-49% ~30%, >=50% ~30%
    "pdl1_tps": {"<1": 0.40, "1-49": 0.30, ">=50": 0.30},
}

BREAST_BIOMARKERS = {
    "her2_positive": 0.18,           # IHC 3+ or ISH amplified
    "her2_low": 0.40,                # IHC 1+ or 2+/ISH-
    "er_positive": 0.75,
    "pr_positive": 0.65,
    "brca1_germline": 0.025,
    "brca2_germline": 0.030,
    "pik3ca_mutation": 0.35,
}

CRC_BIOMARKERS = {
    "kras_mutation": 0.42,
    "nras_mutation": 0.04,
    "braf_v600e": 0.10,
    "her2_amplified": 0.04,
    "msi_high": 0.15,                # higher in early stage, lower metastatic
    "ntrk_fusion": 0.003,
}

GASTRIC_BIOMARKERS = {
    "her2_positive": 0.17,
    "msi_high": 0.08,
    "pdl1_cps_ge5": 0.55,
    "claudin_18_2_positive": 0.38,
}

OVARIAN_BIOMARKERS = {
    "brca1_2_mutation": 0.18,         # combined germline + somatic
    "hrd_positive": 0.50,
    "folr1_high": 0.35,
}

# ECOG distribution among oncology trial candidates (most trials require 0-1)
ECOG_DISTRIBUTION = {
    0: 0.35,
    1: 0.45,
    2: 0.15,
    3: 0.04,
    4: 0.01,
}

# Prior treatment line distribution (varies wildly; this is a rough mix)
PRIOR_LINES_DISTRIBUTION = {
    0: 0.30,   # treatment-naive
    1: 0.35,
    2: 0.20,
    3: 0.10,
    4: 0.05,
}


@dataclass
class BiomarkerProfile:
    """Molecular and clinical profile attached to a synthetic oncology patient."""

    cancer_type: CancerType
    stage: str                                  # 'I', 'II', 'III', 'IV'
    histology: str | None = None
    ecog: int = 1
    prior_lines: int = 0
    prior_therapy_classes: list[str] = field(default_factory=list)
    mutations: dict[str, bool] = field(default_factory=dict)
    expression: dict[str, Any] = field(default_factory=dict)
    other: dict[str, Any] = field(default_factory=dict)
    cns_metastases: bool = False
    measurable_disease: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_dict_distribution(
    dist: dict[Any, float], rng: random.Random
) -> Any:
    """Sample a key from a {key: probability} mapping. Probabilities should sum ~1."""
    keys = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def _sample_independent_mutations(
    table: dict[str, float], rng: random.Random
) -> dict[str, bool]:
    return {marker: rng.random() < p for marker, p in table.items()}


# ---------------------------------------------------------------------------
# Per-cancer-type generators
# ---------------------------------------------------------------------------


def generate_nsclc_profile(rng: random.Random) -> BiomarkerProfile:
    mutations = _sample_independent_mutations(NSCLC_BIOMARKERS, rng)
    # Crude mutual-exclusivity adjustment: keep at most one driver
    drivers = [
        "egfr_mutation",
        "kras_g12c",
        "kras_other",
        "alk_rearrangement",
        "ros1_rearrangement",
        "braf_v600e",
        "met_exon14",
        "ret_fusion",
    ]
    active_drivers = [d for d in drivers if mutations.get(d)]
    if len(active_drivers) > 1:
        keep = rng.choice(active_drivers)
        for d in active_drivers:
            if d != keep:
                mutations[d] = False

    pdl1_bucket = _sample_dict_distribution(NSCLC_EXPRESSION["pdl1_tps"], rng)

    histology = rng.choices(
        ["adenocarcinoma", "squamous", "large_cell", "nsclc_nos"],
        weights=[0.60, 0.30, 0.05, 0.05],
        k=1,
    )[0]

    stage = rng.choices(["I", "II", "III", "IV"], weights=[0.10, 0.10, 0.30, 0.50])[0]

    return BiomarkerProfile(
        cancer_type="nsclc",
        stage=stage,
        histology=histology,
        ecog=_sample_dict_distribution(ECOG_DISTRIBUTION, rng),
        prior_lines=_sample_dict_distribution(PRIOR_LINES_DISTRIBUTION, rng),
        mutations=mutations,
        expression={"pdl1_tps_bucket": pdl1_bucket},
        cns_metastases=rng.random() < 0.25,
        measurable_disease=rng.random() < 0.90,
    )


def generate_breast_profile(rng: random.Random) -> BiomarkerProfile:
    # HER2 / hormone receptor status sampled jointly to be a bit more realistic
    her2 = rng.choices(["positive", "low", "negative"], weights=[0.18, 0.40, 0.42])[0]
    er = rng.random() < 0.75
    pr = rng.random() < 0.65 if er else rng.random() < 0.10
    triple_negative = (not er) and (not pr) and her2 == "negative"

    mutations = {
        "brca1_germline": rng.random() < BREAST_BIOMARKERS["brca1_germline"],
        "brca2_germline": rng.random() < BREAST_BIOMARKERS["brca2_germline"],
        "pik3ca_mutation": rng.random() < BREAST_BIOMARKERS["pik3ca_mutation"],
    }

    stage = rng.choices(["I", "II", "III", "IV"], weights=[0.25, 0.35, 0.25, 0.15])[0]

    return BiomarkerProfile(
        cancer_type="breast",
        stage=stage,
        ecog=_sample_dict_distribution(ECOG_DISTRIBUTION, rng),
        prior_lines=_sample_dict_distribution(PRIOR_LINES_DISTRIBUTION, rng),
        mutations=mutations,
        expression={"er_positive": er, "pr_positive": pr, "her2_status": her2},
        other={"triple_negative": triple_negative},
        cns_metastases=rng.random() < 0.10,
        measurable_disease=rng.random() < 0.85,
    )


def generate_crc_profile(rng: random.Random) -> BiomarkerProfile:
    mutations = _sample_independent_mutations(CRC_BIOMARKERS, rng)
    # KRAS and NRAS are mutually exclusive
    if mutations.get("kras_mutation") and mutations.get("nras_mutation"):
        mutations["nras_mutation"] = False

    stage = rng.choices(["II", "III", "IV"], weights=[0.20, 0.30, 0.50])[0]

    return BiomarkerProfile(
        cancer_type="colorectal",
        stage=stage,
        histology="adenocarcinoma",
        ecog=_sample_dict_distribution(ECOG_DISTRIBUTION, rng),
        prior_lines=_sample_dict_distribution(PRIOR_LINES_DISTRIBUTION, rng),
        mutations=mutations,
        cns_metastases=rng.random() < 0.05,
        measurable_disease=rng.random() < 0.90,
    )


def generate_gastric_profile(rng: random.Random) -> BiomarkerProfile:
    mutations = _sample_independent_mutations(GASTRIC_BIOMARKERS, rng)
    stage = rng.choices(["II", "III", "IV"], weights=[0.15, 0.30, 0.55])[0]
    return BiomarkerProfile(
        cancer_type="gastric",
        stage=stage,
        ecog=_sample_dict_distribution(ECOG_DISTRIBUTION, rng),
        prior_lines=_sample_dict_distribution(PRIOR_LINES_DISTRIBUTION, rng),
        mutations=mutations,
        cns_metastases=rng.random() < 0.05,
        measurable_disease=rng.random() < 0.85,
    )


def generate_ovarian_profile(rng: random.Random) -> BiomarkerProfile:
    mutations = _sample_independent_mutations(OVARIAN_BIOMARKERS, rng)
    stage = rng.choices(["I", "II", "III", "IV"], weights=[0.15, 0.20, 0.50, 0.15])[0]
    return BiomarkerProfile(
        cancer_type="ovarian",
        stage=stage,
        histology="serous",
        ecog=_sample_dict_distribution(ECOG_DISTRIBUTION, rng),
        prior_lines=_sample_dict_distribution(PRIOR_LINES_DISTRIBUTION, rng),
        mutations=mutations,
        cns_metastases=rng.random() < 0.03,
        measurable_disease=rng.random() < 0.85,
    )


GENERATORS = {
    "nsclc": generate_nsclc_profile,
    "breast": generate_breast_profile,
    "colorectal": generate_crc_profile,
    "gastric": generate_gastric_profile,
    "ovarian": generate_ovarian_profile,
}


# ---------------------------------------------------------------------------
# Top-level cohort generator
# ---------------------------------------------------------------------------

# Approximate US cancer incidence shares (only for the types we generate)
DEFAULT_INCIDENCE_WEIGHTS = {
    "nsclc": 0.30,
    "breast": 0.30,
    "colorectal": 0.20,
    "gastric": 0.07,
    "ovarian": 0.08,
    # Rare cancers — under-represented in incidence but key for evaluation
    "cholangiocarcinoma": 0.02,
    "mesothelioma": 0.01,
    "sarcoma": 0.02,
}


def generate_cohort(
    n: int = 1000,
    seed: int = 42,
    weights: dict[str, float] | None = None,
    oversample_rare: bool = True,
) -> list[BiomarkerProfile]:
    """Generate a cohort of synthetic oncology patients.

    Args:
        n: total patients.
        seed: for reproducibility.
        weights: cancer-type mixture. Defaults to DEFAULT_INCIDENCE_WEIGHTS.
        oversample_rare: if True, boosts rare cancers to ~5% each so we have
            enough hard cases for evaluation. Document this if you publish.
    """
    rng = random.Random(seed)
    weights = dict(weights or DEFAULT_INCIDENCE_WEIGHTS)

    if oversample_rare:
        for rare in ["cholangiocarcinoma", "mesothelioma", "sarcoma"]:
            weights[rare] = max(weights.get(rare, 0.0), 0.05)

    # Renormalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    cohort: list[BiomarkerProfile] = []
    for _ in range(n):
        cancer_type = rng.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]
        generator = GENERATORS.get(cancer_type)
        if generator is None:
            # Rare cancers without dedicated generators get a minimal profile
            cohort.append(
                BiomarkerProfile(
                    cancer_type=cancer_type,  # type: ignore[arg-type]
                    stage=rng.choice(["II", "III", "IV"]),
                    ecog=_sample_dict_distribution(ECOG_DISTRIBUTION, rng),
                    prior_lines=_sample_dict_distribution(PRIOR_LINES_DISTRIBUTION, rng),
                )
            )
        else:
            cohort.append(generator(rng))
    return cohort


def save_cohort(cohort: list[BiomarkerProfile], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([p.to_dict() for p in cohort], f, indent=2)
    logger.info("Saved %d profiles to %s", len(cohort), path)
    return path


__all__ = [
    "BiomarkerProfile",
    "generate_cohort",
    "save_cohort",
    "DEFAULT_INCIDENCE_WEIGHTS",
]
