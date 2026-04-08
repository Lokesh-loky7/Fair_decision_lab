# Exports FairAction, FairObservation, FairDecisionEnv for external use.
# Lazy imports to avoid circular import issues during grader/scenario testing.

def __getattr__(name):
    if name == "FairAction":
        from fair_decision_lab.models import FairAction
        return FairAction
    if name == "FairObservation":
        from fair_decision_lab.models import FairObservation
        return FairObservation
    if name == "FairDecisionEnv":
        from fair_decision_lab.server.environment import FairDecisionEnv
        return FairDecisionEnv
    raise AttributeError(f"module 'fair_decision_lab' has no attribute {name!r}")

__all__ = ["FairAction", "FairObservation", "FairDecisionEnv"]
