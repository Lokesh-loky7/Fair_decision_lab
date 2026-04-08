# Generates synthetic bias scenarios for FairDecisionLab.
# true_gap stored as proportion (0.0-1.0). Never appears in summary string.

import random
from dataclasses import dataclass, field
from typing import List

DOMAINS = [
    "hiring",
    "lending",
    "university admissions",
    "healthcare resource allocation",
    "parole decisions",
]


@dataclass
class Scenario:
    task_id: str
    summary: str
    group_labels: List[str]
    true_gap: float
    true_bias_exists: bool
    domain: str
    true_bias: bool = True
    max_turns: int = 1


def _fmt(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _pick_domain(rng: random.Random) -> str:
    return rng.choice(DOMAINS)


def _outcome_for(domain: str) -> str:
    return {
        "hiring": "job offer rate",
        "lending": "loan approval rate",
        "university admissions": "admission rate",
        "healthcare resource allocation": "treatment referral rate",
        "parole decisions": "parole grant rate",
    }[domain]


def generate_easy(seed=None) -> Scenario:
    rng = random.Random(seed)
    domain = _pick_domain(rng)
    outcome = _outcome_for(domain)
    high = round(rng.uniform(0.65, 0.90), 3)
    gap  = round(rng.uniform(0.25, 0.65), 3)
    low  = round(max(high - gap, 0.05), 3)
    true_gap = round(high - low, 3)
    group_a, group_b = "Group A", "Group B"
    summary = (
        f"Domain: {domain.title()}.\n"
        f"Dataset: {rng.randint(5, 15) * 1000:,} records. Outcome measured: {outcome}.\n"
        f"Two demographic groups are present in the data.\n"
        f"  {group_a}: {_fmt(high)}\n"
        f"  {group_b}: {_fmt(low)}\n"
        f"No other covariates are provided."
    )
    return Scenario(
        task_id="easy",
        summary=summary,
        group_labels=[group_a, group_b],
        true_gap=true_gap,
        true_bias=True,
        true_bias_exists=True,
        domain=domain,
        max_turns=1,
    )


def generate_medium(seed=None) -> Scenario:
    rng = random.Random(seed)
    domain = _pick_domain(rng)
    outcome = _outcome_for(domain)
    high = round(rng.uniform(0.40, 0.65), 3)
    gap  = round(rng.uniform(0.08, 0.18), 3)
    low  = round(max(high - gap, 0.10), 3)
    true_gap = round(high - low, 3)
    true_bias = true_gap > 0.10
    overall = round((high + low) / 2, 3)
    group_a, group_b = "Group A", "Group B"
    summary = (
        f"Domain: {domain.title()}.\n"
        f"Dataset: {rng.randint(6, 12) * 1000:,} records. Outcome measured: {outcome}.\n"
        f"Overall {outcome}: {_fmt(overall)}.\n"
        f"Two demographic groups are present in the data.\n"
        f"  {group_a}: {_fmt(high)}\n"
        f"  {group_b}: {_fmt(low)}\n"
        f"Applicant qualifications are reported as broadly similar across groups."
    )
    return Scenario(
        task_id="medium",
        summary=summary,
        group_labels=[group_a, group_b],
        true_gap=true_gap,
        true_bias=true_bias,
        true_bias_exists=true_bias,
        domain=domain,
        max_turns=1,
    )


def generate_hard(seed=None) -> Scenario:
    rng = random.Random(seed)
    domain = _pick_domain(rng)
    outcome = _outcome_for(domain)
    rates = {
        ("A", "X"): round(rng.uniform(0.60, 0.80), 3),
        ("A", "Y"): round(rng.uniform(0.50, 0.70), 3),
        ("B", "X"): round(rng.uniform(0.35, 0.55), 3),
        ("B", "Y"): round(rng.uniform(0.15, 0.35), 3),
    }
    all_rates = list(rates.values())
    true_gap = round(max(all_rates) - min(all_rates), 3)
    group_labels = [f"Group {v1}+{v2}" for (v1, v2) in rates]
    lines = [
        f"Domain: {domain.title()}.",
        f"Dataset: {rng.randint(8, 15) * 1000:,} records. Outcome measured: {outcome}.",
        f"Two protected attributes are present: Attribute-1 (A/B) and Attribute-2 (X/Y).",
        f"Subgroup {outcome}s:",
    ]
    for (v1, v2), rate in rates.items():
        lines.append(f"  Attribute-1={v1}, Attribute-2={v2}: {_fmt(rate)}")
    lines.append("No other covariates are provided.")
    return Scenario(
        task_id="hard",
        summary="\n".join(lines),
        group_labels=group_labels,
        true_gap=true_gap,
        true_bias=True,
        true_bias_exists=True,
        domain=domain,
        max_turns=3,
    )
