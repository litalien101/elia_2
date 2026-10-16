"""Cooperating CPU-first cognitive subsystems used by ELIA."""

from .consciousness import ConsciousnessController
from .consolidation import BackgroundConsolidator, ConsolidationSummary
from .planning import TaskPlanner
from .reflection import ReflectiveAgent, ReflectionReport
from .self_model import SelfModel
from .state_controller import StateController
from .system import CognitiveSystem
from .workspace import ActiveWorkspace

__all__ = [
    "ActiveWorkspace",
    "BackgroundConsolidator",
    "CognitiveSystem",
    "ConsciousnessController",
    "ConsolidationSummary",
    "ReflectiveAgent",
    "ReflectionReport",
    "SelfModel",
    "StateController",
    "TaskPlanner",
]
