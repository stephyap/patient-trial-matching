"""
ClinicalTrials.gov API v2 client for oncology trial retrieval.

The v2 API is REST/JSON, requires no auth, and is rate-limited to ~10 req/sec.
Docs: https://clinicaltrials.gov/data-api/api

Design notes:
- We paginate via `pageToken`, not offset (the v2 way).
- We retry on 5xx and 429 with exponential backoff.
- Eligibility text is preserved as-is — parsing it is the *project's whole point*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_PAGE_SIZE = 100  # API max is 1000, but smaller pages are kinder on memory
DEFAULT_TIMEOUT = 30.0


# Fields we actually need. Keeping the projection tight cuts response size ~5x.
DEFAULT_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "Phase",
    "StudyType",
    "Condition",
    "BriefSummary",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "HealthyVolunteers",
    "StdAge",
    "InterventionType",
    "InterventionName",
    "LocationCountry",
    "LocationState",
    "LocationCity",
    "LocationFacility",
    "PrimaryCompletionDate",
    "LeadSponsorName",
]


@dataclass
class TrialQuery:
    """Structured query for ClinicalTrials.gov v2.

    The `advanced` field uses Essie syntax — see
    https://clinicaltrials.gov/find-studies/constructing-complex-search-queries
    """

    condition: str | None = None              # query.cond
    intervention: str | None = None           # query.intr
    title: str | None = None                  # query.titles
    location: str | None = None               # query.locn
    advanced: str | None = None               # filter.advanced (Essie syntax)
    overall_status: list[str] = field(default_factory=lambda: ["RECRUITING"])
    fields: list[str] = field(default_factory=lambda: DEFAULT_FIELDS)
    page_size: int = DEFAULT_PAGE_SIZE

    def to_params(self, page_token: str | None = None) -> dict[str, str]:
        params: dict[str, str] = {
            "format": "json",
            "pageSize": str(self.page_size),
            "countTotal": "true",
            "fields": "|".join(self.fields),
        }
        if self.condition:
            params["query.cond"] = self.condition
        if self.intervention:
            params["query.intr"] = self.intervention
        if self.title:
            params["query.titles"] = self.title
        if self.location:
            params["query.locn"] = self.location
        if self.advanced:
            params["filter.advanced"] = self.advanced
        if self.overall_status:
            params["filter.overallStatus"] = ",".join(self.overall_status)
        if page_token:
            params["pageToken"] = page_token
        return params


class ClinicalTrialsClient:
    """Thin, retrying client for ClinicalTrials.gov v2."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = "patient-trial-matching/0.1 (research)",
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )

    def __enter__(self) -> "ClinicalTrialsClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        resp = self._client.get(self.base_url, params=params)
        # Retry on 429/5xx, fail fast on 4xx
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.raise_for_status()
        if resp.status_code >= 400:
            logger.error("Bad request: %s %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        return resp.json()

    def iter_studies(self, query: TrialQuery) -> Iterator[dict[str, Any]]:
        """Yield study dicts one at a time, paginating transparently."""
        page_token: str | None = None
        first_page = True
        total: int | None = None
        seen = 0
        pbar: tqdm | None = None

        while True:
            data = self._get(query.to_params(page_token=page_token))
            studies = data.get("studies", [])

            if first_page:
                total = data.get("totalCount")
                logger.info("Total matching studies: %s", total)
                if total:
                    pbar = tqdm(total=total, desc="trials", unit="trial")
                first_page = False

            for study in studies:
                yield study
                seen += 1
                if pbar is not None:
                    pbar.update(1)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        if pbar is not None:
            pbar.close()
        logger.info("Retrieved %d studies", seen)

    def fetch_studies(self, query: TrialQuery, limit: int | None = None) -> list[dict[str, Any]]:
        """Fetch all (or up to `limit`) studies for a query into a list."""
        out: list[dict[str, Any]] = []
        for study in self.iter_studies(query):
            out.append(study)
            if limit is not None and len(out) >= limit:
                break
        return out


# ---------------------------------------------------------------------------
# Convenience: oncology-focused queries
# ---------------------------------------------------------------------------


def oncology_query(
    conditions: list[str] | None = None,
    phases: tuple[str, ...] = ("PHASE2", "PHASE3"),
    country: str = "United States",
    interventional_only: bool = True,
) -> TrialQuery:
    """Build a query for recruiting oncology trials.

    Default focus: solid tumors with rich biomarker variation.
    """
    if conditions is None:
        conditions = [
            "non-small cell lung cancer",
            "breast cancer",
            "colorectal cancer",
            "gastric cancer",
            "pancreatic cancer",
            "ovarian cancer",
            # rare cancers (smaller N but biomarker-rich)
            "cholangiocarcinoma",
            "mesothelioma",
            "soft tissue sarcoma",
        ]

    # OR-join conditions, restrict to interventional + chosen phases + US sites
    cond_clause = " OR ".join(f'"{c}"' for c in conditions)
    advanced_parts = [f"AREA[Condition]({cond_clause})"]
    if phases:
        phase_clause = " OR ".join(phases)
        advanced_parts.append(f"AREA[Phase]({phase_clause})")
    if interventional_only:
        advanced_parts.append("AREA[StudyType]INTERVENTIONAL")
    if country:
        advanced_parts.append(f'AREA[LocationCountry]"{country}"')

    return TrialQuery(
        advanced=" AND ".join(advanced_parts),
        overall_status=["RECRUITING"],
    )


# ---------------------------------------------------------------------------
# Flattening: API returns nested protocolSection -> Parquet-friendly rows
# ---------------------------------------------------------------------------


def flatten_study(study: dict[str, Any]) -> dict[str, Any]:
    """Flatten a study record into a row for parquet storage."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    cond_mod = proto.get("conditionsModule", {})
    elig = proto.get("eligibilityModule", {})
    desc = proto.get("descriptionModule", {})
    interv = proto.get("armsInterventionsModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})

    locations = contacts.get("locations") or []
    interventions = interv.get("interventions") or []

    return {
        "nct_id": ident.get("nctId"),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "overall_status": status_mod.get("overallStatus"),
        "primary_completion_date": (
            status_mod.get("primaryCompletionDateStruct", {}).get("date")
        ),
        "phases": design.get("phases") or [],
        "study_type": design.get("studyType"),
        "conditions": cond_mod.get("conditions") or [],
        "brief_summary": desc.get("briefSummary"),
        "eligibility_criteria": elig.get("eligibilityCriteria"),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "std_ages": elig.get("stdAges") or [],
        "intervention_types": [i.get("type") for i in interventions],
        "intervention_names": [i.get("name") for i in interventions],
        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name"),
        "n_locations": len(locations),
        "locations_us": [
            {
                "facility": loc.get("facility"),
                "city": loc.get("city"),
                "state": loc.get("state"),
                "country": loc.get("country"),
            }
            for loc in locations
            if loc.get("country") == "United States"
        ],
    }


def studies_to_parquet(studies: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a flattened DataFrame to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_study(s) for s in studies]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d trials to %s", len(df), path)
    return path


__all__ = [
    "ClinicalTrialsClient",
    "TrialQuery",
    "DEFAULT_FIELDS",
    "oncology_query",
    "flatten_study",
    "studies_to_parquet",
]
