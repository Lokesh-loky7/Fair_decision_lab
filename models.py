# Data models for FairDecisionLab.
# Imports from openenv.core.env_server.types — never from pydantic BaseModel directly.

from typing import Optional, List
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class FairAction(Action):
    analysis: str = Field(
        description="The agent's free-text reasoning about the dataset and any bias present."
    )
    identified_bias: bool = Field(
        description="True if the agent believes bias exists in the dataset."
    )
    group_gap_estimate: Optional[float] = Field(
        default=None,
        description="Agent's numeric estimate of the approval rate gap between groups, in percentage points."
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Agent's recommended corrective action. One of: reweight, remove_feature, none."
    )


class FairObservation(Observation):
    task_id: str = Field(
        description="Unique identifier for the current task/episode."
    )
    step: int = Field(
        description="Current step number within the episode (1-indexed)."
    )
    dataset_summary: str = Field(
        description="Natural language summary of the synthetic dataset shown to the agent."
    )
    group_labels: List[str] = Field(
        default_factory=list,
        description="List of group label names present in the dataset (e.g. ['Group A', 'Group B'])."
    )
    feedback: str = Field(
        default="",
        description="Grader feedback from the previous turn. Empty string on the first turn."
    )
    turn: int = Field(
        default=0,
        description="Current turn number within a multi-turn episode (0-indexed)."
    )
    max_turns: int = Field(
        default=3,
        description="Total number of turns allowed for this episode."
    )
