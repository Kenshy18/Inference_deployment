from __future__ import annotations

import pytest

from production.polygon.runtime.optimizer_adapters import resources


def test_transport_defaults_to_existing_fork_behavior(monkeypatch) -> None:
    monkeypatch.delenv(resources.WORKER_START_METHOD_ENV, raising=False)
    assert resources.selected_worker_transport() == "fork"


@pytest.mark.parametrize("value", ("spawn", "fork"))
def test_transport_accepts_supported_method(monkeypatch, value: str) -> None:
    monkeypatch.setenv(resources.WORKER_START_METHOD_ENV, value)
    assert resources.selected_worker_transport() == value


def test_transport_rejects_unknown_method(monkeypatch) -> None:
    monkeypatch.setenv(resources.WORKER_START_METHOD_ENV, "forkserver")
    with pytest.raises(ValueError, match=resources.WORKER_START_METHOD_ENV):
        resources.selected_worker_transport()
