#!/usr/bin/env python3
"""Synthetic state-machine smoke test; contains no trained model or visual evaluation."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from ocr_edr import Action, Observation, Rendered, RepairLoop, Verdict
from ocr_edr.loop import digest


class SyntheticRenderer:
    def render(self, observation):
        return Rendered(
            "synthetic://" + digest(observation.prediction),
            digest(observation.prediction),
            "synthetic_hash",
        )


class ScriptedJudge:
    def __init__(self, expected):
        self.expected = expected

    def assess(self, observation, rendered):
        return Verdict(
            "good" if observation.prediction == self.expected else "bad", 1.0, "scripted_demo"
        )


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def act(self, observation, rendered, verdict, history):
        return next(self.actions, Action("stop"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/runs/demo/trace.jsonl"))
    args = parser.parse_args()
    cases = [
        (
            "formula_repair",
            "formula",
            "x^3",
            "x^2",
            [Action("patch", "3", "2"), Action("request_render")],
        ),
        ("formula_preserve", "formula", r"x \geq 0", r"x \geq 0", []),
        (
            "table_repair",
            "table",
            "<table><tr><td>13</td></tr></table>",
            "<table><tr><td>12</td></tr></table>",
            [Action("patch", "13", "12"), Action("request_render")],
        ),
        (
            "stale_render_rollback",
            "formula",
            "y^3",
            "y^2",
            [Action("patch", "3", "2"), Action("stop")],
        ),
    ]
    outcomes = []
    for name, modality, initial, expected, actions in cases:
        outcome = RepairLoop(
            SyntheticRenderer(), ScriptedJudge(expected), ScriptedPolicy(actions)
        ).run(Observation(name, modality, "synthetic://source", initial))
        outcomes.append({"synthetic": True, **outcome.to_dict()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(o) + "\n" for o in outcomes), encoding="utf-8")
    print(json.dumps({o["sample_id"]: o["status"] for o in outcomes}, indent=2))


if __name__ == "__main__":
    main()
