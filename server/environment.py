# Core RL environment for FairDecisionLab.

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import FairAction, FairObservation
from server.scenario_generator import generate_easy, generate_medium, generate_hard
from server.grader import grade_easy, grade_medium, grade_hard

_GENERATORS = {"easy": generate_easy, "medium": generate_medium, "hard": generate_hard}
_MAX_TURNS  = {"easy": 1, "medium": 1, "hard": 3}


class FairDecisionEnvironment(Environment):

    def __init__(self):
        self._scenario   = None
        self._history    = []
        self._task_id    = "easy"
        self._step       = 0
        self._turn       = 0
        self._done       = False
        self._state      = State(step=0, done=False)

    def reset(self, task_id: str = "easy", seed=None) -> FairObservation:
        if task_id not in _GENERATORS:
            raise ValueError(f"Unknown task_id: {task_id!r}")
        self._task_id    = task_id
        self._scenario   = _GENERATORS[task_id](seed=seed)
        self._history    = []
        self._step       = 0
        self._turn       = 0
        self._done       = False
        self._state      = State(step=0, done=False)
        return FairObservation(
            task_id         = task_id,
            step            = self._step,
            dataset_summary = self._scenario.summary,
            group_labels    = self._scenario.group_labels,
            feedback        = "",
            turn            = self._turn,
            max_turns       = _MAX_TURNS[task_id],
            done            = False,
            reward          = None,
        )

    def step(self, action: FairAction):
        if self._scenario is None:
            raise RuntimeError("Call reset() before step().")
        self._step += 1

        if self._task_id == "easy":
            reward = grade_easy(action, self._scenario)
        elif self._task_id == "medium":
            reward = grade_medium(action, self._scenario)
        else:
            reward = grade_hard(action, self._scenario,
                                turn=self._turn,
                                history=[a.analysis for a in self._history])

        self._history.append(action)

        if self._task_id == "hard":
            self._turn += 1
            done = self._turn >= 3
        else:
            done = True

        self._done  = done
        self._state = State(step=self._step, done=done)

        obs = FairObservation(
            task_id         = self._task_id,
            step            = self._step,
            dataset_summary = self._scenario.summary,
            group_labels    = self._scenario.group_labels,
            feedback        = f"Step reward: {reward:.4f}",
            turn            = self._turn,
            max_turns       = _MAX_TURNS[self._task_id],
            done            = done,
            reward          = reward,
        )
        return obs

    @property
    def state(self) -> State:
        return self._state