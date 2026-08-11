"""Time-window logic for deciding when a person alert should fire."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class WindowStats:
    """Summary of the current rolling detection window."""

    samples: int
    positive_samples: int
    required_samples: int
    window_size: int

    @property
    def ratio(self) -> float:
        return self.positive_samples / self.samples if self.samples else 0.0

    @property
    def ready(self) -> bool:
        return self.samples >= self.window_size

    @property
    def qualifies(self) -> bool:
        return self.ready and self.positive_samples >= self.required_samples


class DetectionWindow:
    """Keep the last N one-second person/no-person samples."""

    def __init__(self, window_size: int, minimum_ratio: float) -> None:
        if window_size < 1:
            raise ValueError("window_size 必須至少為 1")
        if not 0 < minimum_ratio <= 1:
            raise ValueError("minimum_ratio 必須大於 0 且小於等於 1")

        self.window_size = window_size
        self.required_samples = max(1, ceil(window_size * minimum_ratio))
        self._samples: deque[bool] = deque(maxlen=window_size)

    def add(self, person_detected: bool) -> WindowStats:
        self._samples.append(bool(person_detected))
        positive_samples = sum(self._samples)
        return WindowStats(
            samples=len(self._samples),
            positive_samples=positive_samples,
            required_samples=self.required_samples,
            window_size=self.window_size,
        )

    def stats(self) -> WindowStats:
        """Return the current window without adding a new sample."""

        return WindowStats(
            samples=len(self._samples),
            positive_samples=sum(self._samples),
            required_samples=self.required_samples,
            window_size=self.window_size,
        )


class AlertGate:
    """Prevent repeated alerts while the same person event remains active."""

    def __init__(self, cooldown_seconds: float = 300.0) -> None:
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._active = False
        self._last_alert_at: float | None = None

    def should_alert(self, stats: WindowStats, now: float) -> bool:
        if not stats.qualifies:
            # A new alert is allowed after the rolling window no longer
            # qualifies, meaning the person event has ended.
            self._active = False
            return False

        if self._active:
            return False

        if (
            self._last_alert_at is not None
            and now - self._last_alert_at < self.cooldown_seconds
        ):
            return False

        self._active = True
        self._last_alert_at = now
        return True
