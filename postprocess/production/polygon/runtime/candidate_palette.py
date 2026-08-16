"""Frozen class-specific multistate DP candidate palette by target interval."""

from __future__ import annotations

from .candidate_config import CANDIDATE, LABELS, CandidateConfig


_BASELINE = ("C02_125", "G02", "G04", "A06", "F3_P1", "D6_P1")
_MALE = ("C02_125", "G02_H3", "G04_H3", "A06_K3", "F3_P1", "D6_R5_P1")


def role_ids(
    label: str,
    target_interval: int = 6,
    config: CandidateConfig = CANDIDATE,
) -> tuple[str, ...]:
    """Return non-raw states; raw is the implicit state zero."""
    config.validate()
    if label not in LABELS:
        raise ValueError(f"unsupported label: {label!r}")
    if int(target_interval) != int(config.temporal.target_interval):
        raise ValueError(
            f"{config.profile_id} is frozen at target interval "
            f"{config.temporal.target_interval}"
        )
    interval = int(target_interval)
    if label == "女性器":
        return _BASELINE if interval < 2 else _BASELINE + ("F3_Q75_P1",)
    if label == "男性器":
        return _MALE
    return _BASELINE if interval < 4 else _BASELINE + ("VF8_P1",)
