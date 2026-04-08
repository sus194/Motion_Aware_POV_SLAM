"""World container."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.sim.objects import SimObject


@dataclass
class World:
    width: float
    height: float
    objects: list[SimObject] = field(default_factory=list)
    walls: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.width, self.height)
