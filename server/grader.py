import re
from models import FairAction
from server.scenario_generator import Scenario


def clamp(x: float) -> float:
    return max(0.05, min(0.95, x))


def extract_number(text: str):
    """Return first number as proportion (0.0-1.0)."""
    match = re.search(r"(\d{1,3}(?:\.\d{1,2})?)\s*%|(?<!\d)(0?\.\d{1,2})(?!\d)", text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    val = float(raw)
    if val > 1.0:
        val = val / 100.0
    return round(val, 4)


def _to_proportion(val):
    """Normalise to proportion. Handles both proportion (0.40) and pp (40.0) inputs."""
    if val is None:
        return None
    return val / 100.0 if val > 1.0 else val


def bias_mentioned(text: str) -> bool:
    return bool(re.search(
        r"bias|discriminat|unfair|disparity|gap|inequit|imbalance|disparate",
        text, re.IGNORECASE
    ))


def mitigation_mentioned(text: str):
    t = text.lower()
    if "reweight" in t or "re-weight" in t:
        return "reweight"
    if "remove" in t or "drop" in t:
        return "remove_feature"
    if "oversample" in t or "augment" in t:
        return "oversample"
    return None


def grade_easy(action: FairAction, scenario: Scenario) -> float:
    # All comparisons in proportion space
    c1 = 0.40 if action.identified_bias == scenario.true_bias else 0.0

    c2 = 0.0
    est = _to_proportion(action.group_gap_estimate)
    if est is None:
        est = extract_number(action.analysis)
    if est is not None:
        diff = abs(est - scenario.true_gap)
        if diff <= 0.03:
            c2 = 0.35
        elif diff <= 0.08:
            c2 = 0.22
        elif diff <= 0.15:
            c2 = 0.12
        else:
            c2 = 0.03

    words = len(action.analysis.split())
    c3 = (0.15 if bias_mentioned(action.analysis) else 0.0) + (0.10 if words > 30 else 0.0)
    return clamp(c1 + c2 + c3)


def grade_medium(action: FairAction, scenario: Scenario) -> float:
    c1 = 0.35 if action.identified_bias == scenario.true_bias else 0.0

    c2 = 0.0
    est = _to_proportion(action.group_gap_estimate)
    if est is None:
        est = extract_number(action.analysis)
    if est is not None:
        diff = abs(est - scenario.true_gap)
        if diff <= 0.02:
            c2 = 0.35
        elif diff <= 0.05:
            c2 = 0.22
        elif diff <= 0.10:
            c2 = 0.12
        else:
            c2 = 0.04

    words = len(action.analysis.split())
    depth = 0.12 if words >= 100 else (0.06 if words >= 60 else 0.01)
    text_lower = action.analysis.lower()
    fairness_kw = 0.05 if "fairness" in text_lower else 0.0
    standards = ("80%", "four-fifths", "four fifths", "disparate impact", "demographic parity")
    standards_bonus = 0.13 if any(s in text_lower for s in standards) else 0.0
    c3 = min(depth + fairness_kw + standards_bonus, 0.30)
    return clamp(c1 + c2 + c3)


def grade_hard(action: FairAction, scenario: Scenario, turn: int, history: list) -> float:
    text = action.analysis
    score = 0.0

    if turn == 0 or turn == 1:
        axis1 = bool(re.search(r"attribute.?1|group [ab]|a/b", text, re.IGNORECASE))
        axis2 = bool(re.search(r"attribute.?2|x/y|subgroup", text, re.IGNORECASE))
        if axis1 and axis2:
            score = 0.35
        elif axis1 or axis2:
            score = 0.15
        else:
            score = 0.03

    elif turn == 2:
        est = _to_proportion(action.group_gap_estimate)
        if est is None:
            est = extract_number(text)
        if est is not None:
            diff = abs(est - scenario.true_gap)
            if diff <= 0.03:
                score = 0.35
            elif diff <= 0.08:
                score = 0.15
            else:
                score = 0.08
        else:
            score = 0.05

    elif turn == 3:
        valid = {"reweight", "remove_feature", "oversample"}
        if action.recommended_action and action.recommended_action.lower().replace(" ", "_") in valid:
            score = 0.35
        elif mitigation_mentioned(text) is not None:
            score = 0.20
        elif re.search(r"mitigat|correct|address|fix|reduc", text, re.IGNORECASE):
            score = 0.10
        else:
            score = 0.03

    if history:
        prev = history[-1] if isinstance(history[-1], str) else history[-1].analysis
        if len(text.split()) > len(prev.split()):
            score += 0.05

    return clamp(score)