---
title: Fair Decision Lab
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - fairness
  - bias-detection
---

# FairDecisionLab

## 1. Environment Description & Motivation

FairDecisionLab simulates an AI auditing workflow. An agent receives a synthetic dataset summary — approval rates broken down by demographic group — and must reason in natural language to detect bias, quantify it, and recommend a correction.

AI fairness is a documented real-world problem. Biased models have been shown to discriminate in hiring (Amazon's recruiting tool), lending (mortgage approval disparities), and healthcare (risk-scoring algorithms that systematically underserved Black patients). Auditing these systems requires more than picking a label — it requires structured reasoning about what the numbers mean.

What makes this environment novel is that the agent writes free-text analysis rather than selecting from a menu. The grader evaluates the quality of that reasoning: did the agent identify the right bias axis? Did it estimate the gap accurately? Did it name the correct fairness metric and recommend a concrete mitigation? This makes the environment ungameable — repeating the same action produces the same mediocre score, and a strong agent must demonstrate genuine understanding to score well.

---

## 2. Action Space

| Field                | Type            | Description                                                                                                                |
| -------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `analysis`           | `str`           | Free-text reasoning about the dataset and any bias present. Must be substantive — at least 60 words for full depth credit. |
| `identified_bias`    | `bool`          | `true` if the agent believes bias exists in the dataset, `false` otherwise.                                                |
| `group_gap_estimate` | `float \| null` | Numeric estimate of the approval rate gap between groups, in percentage points.                                            |
| `recommended_action` | `str \| null`   | Recommended corrective action. One of: `reweight`, `remove_feature`, `oversample`, or `null`.                              |

---

## 3. Observation Space

| Field             | Type            | Description                                                                                                              |
| ----------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `task_id`         | `str`           | Unique identifier for the current episode.                                                                               |
| `step`            | `int`           | Current step number within the episode.                                                                                  |
| `dataset_summary` | `str`           | Natural language summary of the synthetic dataset shown to the agent. Never contains ground truth gap or bias label.     |
| `group_labels`    | `list[str]`     | List of demographic group labels present in the dataset.                                                                 |
| `feedback`        | `str`           | Grader feedback from the previous turn. Empty string on the first turn. Used in hard task to guide multi-turn reasoning. |
| `turn`            | `int`           | Current turn number within a multi-turn episode (0-indexed).                                                             |
| `max_turns`       | `int`           | Total turns allowed for this episode. 1 for easy/medium, 3 for hard.                                                     |
| `done`            | `bool`          | Inherited. `true` when the episode has ended.                                                                            |
| `reward`          | `float \| null` | Inherited. Reward from the most recent step, or `null` on reset.                                                         |

---

## 4. Task Descriptions

**Easy** — The dataset contains two demographic groups with a large, obvious approval rate gap of 25–65 percentage points. A competent agent should immediately identify the disparity, estimate the gap within a few percentage points, and name it as disparate impact. This task is designed to be solvable by any model that understands basic fairness concepts. A baseline model should score between 0.75 and 0.90.

**Medium** — The gap is small and ambiguous, between 8 and 18 percentage points. The agent must reason about whether the gap is statistically meaningful, apply the four-fifths rule or demographic parity metric, and justify its conclusion. Simply spotting a difference is not enough — the agent must demonstrate understanding of fairness thresholds. A baseline model should score between 0.45 and 0.65.

**Hard** — A 3-turn episode with intersectional bias across two protected attributes (gender × age group). Turn 1 asks the agent to identify all bias axes. Turn 2 asks it to quantify the worst-case subgroup gap. Turn 3 asks for a specific mitigation strategy. Each turn builds on grader feedback from the previous one. This task is designed to challenge frontier models — a strong model should score between 0.15 and 0.35, and a weak model should score near 0.05.

---

## 5. Baseline Scores

| task_id | avg_reward | num_episodes |
| ------- | ---------- | ------------ |
| easy    | ~0.80      | 3            |
| medium  | ~0.55      | 3            |
| hard    | ~0.25      | 3            |

---

## 6. Setup

```bash
git clone https://huggingface.co/spaces/your-username/fair-decision-lab
pip install -e .
docker build -t fair-decision-lab .
docker run -p 7860:7860 fair-decision-lab
HF_TOKEN=your_token API_BASE_URL=https://api-inference.huggingface.co/v1 MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 ENV_URL=http://localhost:7860 python inference.py
```

---

## 7. Usage Example

```python
from fair_decision_lab.client import FairDecisionEnv
from fair_decision_lab.models import FairAction

env = FairDecisionEnv(base_url="http://localhost:7860")
obs = env.reset(task_id="easy")
action = FairAction(
    analysis="Group A has a 65% approval rate vs 30% for Group B — a 35pp gap indicating disparate impact. I recommend reweighting.",
    identified_bias=True,
    group_gap_estimate=35.0,
    recommended_action="reweight",
)
result = env.step(action)
print(f"reward={result.reward:.4f}  done={result.done}")
```
