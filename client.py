# Client for FairDecisionLab.

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import FairAction, FairObservation


class FairDecisionEnv(EnvClient[FairAction, FairObservation, State]):

    def _step_payload(self, action: FairAction) -> dict:
        return {
            "analysis":           action.analysis,
            "identified_bias":    action.identified_bias,
            "group_gap_estimate": action.group_gap_estimate,
            "recommended_action": action.recommended_action,
        }

    def _parse_result(self, payload: dict) -> StepResult[FairObservation]:
        obs_data = payload.get("observation", {})
        obs = FairObservation(
            task_id         = obs_data.get("task_id", ""),
            step            = obs_data.get("step", 0),
            dataset_summary = obs_data.get("dataset_summary", ""),
            group_labels    = obs_data.get("group_labels", []),
            feedback        = obs_data.get("feedback", ""),
            turn            = obs_data.get("turn", 0),
            max_turns       = obs_data.get("max_turns", 1),
            done            = payload.get("done", False),
            reward          = payload.get("reward"),
        )
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict) -> State:
        return State(
            step = payload.get("step", 0),
            done = payload.get("done", False),
        )