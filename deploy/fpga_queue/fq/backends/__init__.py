"""Execution backends.

A backend turns a job spec into (a) a staging step and (b) a small set of
shell commands, one per lifecycle phase.  It deliberately does **not** run
anything itself -- ``fq/runner.py`` owns process handling, timeouts and
teardown for every backend identically.  That split means the mock backend
exercises the real lifecycle code, so the tests are testing the thing that
ships rather than a parallel implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Backend, JobContext, Phase

if TYPE_CHECKING:  # pragma: no cover
    from ..config import PoolConfig


def get_backend(name: str, options: dict | None = None) -> Backend:
    options = dict(options or {})
    if name == "mock":
        from .mock import MockBackend
        return MockBackend(options)
    if name == "firesim":
        from .firesim import FiresimBackend
        return FiresimBackend(options)
    raise ValueError(f"unknown backend {name!r} (known: firesim, mock)")


__all__ = ["Backend", "JobContext", "Phase", "get_backend"]
