"""Chaos testing module for Strands Evals.

Provides deterministic fault injection for evaluating agent resilience
under tool failures and response corruption scenarios.
"""

from .case import ChaosCase, ChaosEffectsConfig
from .effects import (
    ChaosEffect,
    Confabulation,
    CorruptValues,
    EmptyResponse,
    ExecutionError,
    FullRefusal,
    MalformedJson,
    ModelEffect,
    ModelEffectUnion,
    NetworkError,
    RemoveFields,
    SuccessFraming,
    Timeout,
    ToolEffect,
    ToolEffectUnion,
    TruncateFields,
    ValidationError,
)
from .experiment import ChaosExperiment
from .plugin import ChaosPlugin

__all__ = [
    # Core classes
    "ChaosCase",
    "ChaosEffectsConfig",
    "ChaosExperiment",
    "ChaosPlugin",
    # Effect hierarchy
    "ChaosEffect",
    "ToolEffect",
    "ToolEffectUnion",
    "ModelEffect",
    "ModelEffectUnion",
    # Tool effects
    "Timeout",
    "NetworkError",
    "ExecutionError",
    "ValidationError",
    "TruncateFields",
    "RemoveFields",
    "CorruptValues",
    # Model effects
    "MalformedJson",
    "EmptyResponse",
    "Confabulation",
    "FullRefusal",
    "SuccessFraming",
]
