from __future__ import annotations

from typing import Any, Protocol

import numpy.typing as npt


class RobotPolicy(Protocol):
    """Small interface that every policy backend must implement."""

    def reset(self, task_description: str) -> None:
        """Clear episode state."""

    def act(
        self,
        image: npt.NDArray[Any],
        task_description: str,
        observation: dict[str, npt.NDArray[Any]],
    ) -> npt.NDArray[Any]:
        """Return one LIBERO-compatible 7-DoF action."""

