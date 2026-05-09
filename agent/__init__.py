"""Public exports for the ARC-AGI-3 agent package.

``llm_agent`` is intentionally NOT imported eagerly: it is parked legacy
code that depends on env-driving helpers in ``explorer`` that were removed
during the migration to the real ``arc_agi_3`` SDK. Import it directly
(``from agent import llm_agent``) only if you are working on it.
"""
from .world_model import WorldModel, Transition, ScoreEvent, ObjectTrace
from .explorer import ProbeReport, detect_objects
from .offline_controller import OfflineControllerAgent

__all__ = [
    "WorldModel",
    "Transition",
    "ScoreEvent",
    "ObjectTrace",
    "ProbeReport",
    "detect_objects",
    "OfflineControllerAgent",
]
