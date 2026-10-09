"""Adapter for a team's existing markup-to-PNG rendering executable."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from .loop import Observation, Rendered, digest


class CommandRenderer:
    """Run an argument list with {input}, {output}, {modality}; never invoke a shell.

    Each call gets a new directory. The renderer must write a PNG at {output}.
    A PNG is evidence for the judge, not a consistency verdict by itself.
    """

    def __init__(self, command: list[str], output_root: Path, timeout: float = 30):
        if not command or not any("{input}" in arg for arg in command):
            raise ValueError("Renderer command must include an {input} argument")
        if not any("{output}" in arg for arg in command):
            raise ValueError("Renderer command must include an {output} argument")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Renderer timeout must be positive and finite")
        self.command = command
        self.output_root = output_root.resolve()
        self.timeout = timeout

    def render(self, observation: Observation) -> Rendered:
        self.output_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="render-", dir=self.output_root))
        source = work / ("input.tex" if observation.modality == "formula" else "input.html")
        output = work / "render.png"
        source.write_text(observation.prediction, encoding="utf-8")
        command = [
            arg.format(input=str(source), output=str(output), modality=observation.modality)
            for arg in self.command
        ]
        subprocess.run(command, cwd=work, timeout=self.timeout, check=True, capture_output=True)
        with output.open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Renderer did not produce a PNG")
        return Rendered(str(output), digest(observation.prediction), "external_command")
