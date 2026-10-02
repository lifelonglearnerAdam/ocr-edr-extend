"""Reference-free edit/render/reassess state machine.

A judge's verdict is an inference-time decision, never a benchmark label. References
belong in the offline evaluator and are deliberately absent from Observation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Literal, Protocol


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Observation:
    sample_id: str
    modality: Literal["formula", "table"]
    source_image: str
    prediction: str

    def __post_init__(self) -> None:
        if self.modality not in {"formula", "table"}:
            raise ValueError("Only formula and table observations are supported")


@dataclass(frozen=True)
class Rendered:
    path: str
    prediction_sha256: str
    backend: str


@dataclass(frozen=True)
class Verdict:
    label: Literal["good", "bad", "uncertain"]
    confidence: float
    judge: str

    def __post_init__(self) -> None:
        if self.label not in {"good", "bad", "uncertain"}:
            raise ValueError("Unknown judge label")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Judge confidence must be finite and in [0, 1]")


@dataclass(frozen=True)
class Action:
    kind: Literal["inspect", "diagnose", "patch", "global_patch", "request_render", "stop"]
    find: str = ""
    replacement: str = ""


@dataclass(frozen=True)
class Budget:
    max_steps: int = 6
    max_renders: int = 3  # Includes the initial render.
    accept_confidence: float = 0.9

    def __post_init__(self) -> None:
        for value in (self.max_steps, self.max_renders):
            if type(value) is not int or value < 1:
                raise ValueError("Step and render budgets must be positive integers")
        if not math.isfinite(self.accept_confidence) or not 0 <= self.accept_confidence <= 1:
            raise ValueError("Acceptance confidence must be in [0, 1]")


class Renderer(Protocol):
    def render(self, observation: Observation) -> Rendered: ...


class Judge(Protocol):
    def assess(self, observation: Observation, rendered: Rendered) -> Verdict: ...


class Policy(Protocol):
    def act(
        self,
        observation: Observation,
        rendered: Rendered | None,
        verdict: Verdict | None,
        history: tuple[dict, ...],
    ) -> Action: ...


@dataclass
class Outcome:
    sample_id: str
    initial_prediction: str
    final_prediction: str
    status: str
    steps: int
    renders: int
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RepairLoop:
    def __init__(
        self, renderer: Renderer, judge: Judge, policy: Policy, budget: Budget | None = None
    ):
        self.renderer = renderer
        self.judge = judge
        self.policy = policy
        self.budget = budget or Budget()

    def run(self, initial: Observation) -> Outcome:
        current = initial
        rendered = None
        verdict = None
        renders = steps = 0
        trace: list[dict] = []

        def finish(status: str, accepted: bool = False) -> Outcome:
            return Outcome(
                initial.sample_id,
                initial.prediction,
                current.prediction if accepted else initial.prediction,
                status,
                steps,
                renders,
                trace,
            )

        def render_and_assess() -> None:
            nonlocal rendered, verdict, renders
            renders += 1  # Count failed renderer attempts too.
            rendered = self.renderer.render(current)
            if rendered.prediction_sha256 != digest(current.prediction):
                raise ValueError("Renderer returned evidence for a different prediction")
            verdict = self.judge.assess(current, rendered)
            trace.append(
                {
                    "event": "render_and_assess",
                    "prediction_sha256": digest(current.prediction),
                    "render": asdict(rendered),
                    "verdict": asdict(verdict),
                }
            )

        def good() -> bool:
            return bool(
                rendered is not None
                and rendered.prediction_sha256 == digest(current.prediction)
                and verdict is not None
                and verdict.label == "good"
                and verdict.confidence >= self.budget.accept_confidence
            )

        try:
            render_and_assess()
            if good():
                return finish("preserved", accepted=True)
            for steps in range(1, self.budget.max_steps + 1):
                # Policy receives a copy of history so it cannot mutate the audit trail.
                from copy import deepcopy

                action = self.policy.act(current, rendered, verdict, tuple(deepcopy(trace)))
                trace.append({"event": "action", "step": steps, **asdict(action)})
                if action.kind in {"inspect", "diagnose"}:
                    continue
                if action.kind == "stop":
                    return finish("accepted" if good() else "unverified_rollback", accepted=good())
                if action.kind in {"patch", "global_patch"}:
                    if action.kind == "patch":
                        if not action.find or current.prediction.count(action.find) != 1:
                            return finish("ambiguous_patch_rollback")
                        candidate = current.prediction.replace(action.find, action.replacement, 1)
                    else:
                        candidate = action.replacement
                    current = replace(current, prediction=candidate)
                    rendered = verdict = (
                        None  # Editing invalidates both visual evidence and verdict.
                    )
                    continue
                if action.kind == "request_render":
                    if renders >= self.budget.max_renders:
                        return finish("render_budget_rollback")
                    render_and_assess()
                    if good():
                        return finish("accepted", accepted=True)
                    continue
                return finish("invalid_action_rollback")
            return finish("step_budget_rollback")
        except Exception as error:
            # Exceptions from model/render adapters must not leak a partial edit.
            trace.append({"event": "backend_error", "error_type": type(error).__name__})
            return finish("backend_error_rollback")
