"""
Seismic model: main shock, aftershock schedule, intensity decay, and building damage.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Aftershock:
    step: int
    magnitude: float
    epicenter: Optional[Tuple[float, float]] = None  # None = same as main


@dataclass
class SeismicEvent:
    epicenter: Tuple[float, float]
    magnitude: float
    step: int = 0


class SeismicModel:
    """
    Models seismic impact with exponential decay from epicenter.
    I(r) = I_0 * exp(-k * r)
    where I_0 = 10^(magnitude) scaled, k = decay constant.
    """

    def __init__(self, config: dict, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.epicenter = tuple(config.get('epicenter', [25, 25]))
        self.magnitude = config.get('magnitude', 6.5)
        self.decay_k = config.get('decay_k', 0.05)
        self.intensity_scale = config.get('intensity_scale', 1.0)

        # Aftershock schedule
        self.aftershocks: List[Aftershock] = []
        for asc in config.get('aftershocks', []):
            self.aftershocks.append(Aftershock(
                step=asc['step'],
                magnitude=asc['magnitude'],
                epicenter=tuple(asc['epicenter']) if 'epicenter' in asc else None,
            ))

        # Black swan events (sudden collapse/fire at specific steps)
        self.black_swans: List[dict] = config.get('black_swans', [])

        self.events_log: List[SeismicEvent] = []

    def compute_intensity(self, x: float, y: float,
                          epicenter: Tuple[float, float],
                          magnitude: float) -> float:
        """Compute seismic intensity at point (x,y) given epicenter and magnitude."""
        r = np.sqrt((x - epicenter[0])**2 + (y - epicenter[1])**2)
        I_0 = (10 ** (magnitude - 5)) * self.intensity_scale  # Scale so M6.5 ~ 30
        return I_0 * np.exp(-self.decay_k * r)

    def get_initial_damage(self, grid_width: int, grid_height: int) -> np.ndarray:
        """Compute damage matrix from main shock for all cells."""
        damage = np.zeros((grid_height, grid_width))
        for y in range(grid_height):
            for x in range(grid_width):
                damage[y, x] = self.compute_intensity(
                    x, y, self.epicenter, self.magnitude)
        self.events_log.append(SeismicEvent(self.epicenter, self.magnitude, step=0))
        return damage

    def get_aftershock_damage(self, step: int,
                               grid_width: int,
                               grid_height: int) -> Optional[np.ndarray]:
        """Check if an aftershock occurs at this step; return damage matrix if so."""
        for asc in self.aftershocks:
            if asc.step == step:
                epi = asc.epicenter if asc.epicenter else self.epicenter
                damage = np.zeros((grid_height, grid_width))
                for y in range(grid_height):
                    for x in range(grid_width):
                        damage[y, x] = self.compute_intensity(
                            x, y, epi, asc.magnitude)
                self.events_log.append(SeismicEvent(epi, asc.magnitude, step=step))
                return damage
        return None

    def get_black_swan_events(self, step: int) -> List[dict]:
        """Return any black swan events scheduled for this step."""
        return [bs for bs in self.black_swans if bs.get('step') == step]

    def generate_random_aftershocks(self, main_magnitude: float,
                                     num: int = 3,
                                     start_step: int = 5,
                                     step_spread: int = 10):
        """Auto-generate aftershock schedule based on main magnitude."""
        self.aftershocks = []
        for i in range(num):
            mag = main_magnitude - self.rng.uniform(1.0, 2.5)
            step = start_step + self.rng.integers(0, step_spread)
            offset_x = self.rng.uniform(-10, 10)
            offset_y = self.rng.uniform(-10, 10)
            epi = (self.epicenter[0] + offset_x, self.epicenter[1] + offset_y)
            self.aftershocks.append(Aftershock(step=int(step), magnitude=max(3.0, mag), epicenter=epi))
        self.aftershocks.sort(key=lambda a: a.step)
