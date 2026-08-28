"""How a job is decided to be finished.

There are three mechanisms, and they are tried strictly in this order.  The
ordering matters more than the mechanisms: a job that can end by itself must
never be ended by a timer, because a timer holds an expensive FPGA for its
full duration whether or not the work is done.

1. ``exit`` -- **the fast path, and the default.**
   A well-behaved RISC-V guest terminates the simulation itself.  Zephyr's
   ``sys_reboot(SYS_REBOOT_COLD)`` reaches ``sys_arch_reboot`` in
   ``soc/rocketchip/virt_riscv/common/soc.c``, which writes the standard
   riscv-tests HTIF exit word::

       tohost = (uint64_t)((success << 1) | 1);   // HTIF Exit Command

   ``CONFIG_UART_HTIF=y`` on ``chipyard_riscv64`` (SoC
   ``rocketchip_virt_riscv64``), so this branch is live.  FireSim's TSI bridge
   watches ``tohost``, the simulation stops, ``firesim runworkload`` returns on
   its own, and results are copied back to ``results-workload/`` automatically.
   No polling, no ssh, no guessing.

   **Guests SHOULD call ``sys_reboot()`` when their work is done.**  A guest
   that does not -- the stock Zephyr ``hello_world`` prints and then idles
   forever -- will burn its entire timeout holding an FPGA nobody else can
   use.  That is the single most expensive thing a submitter can do to this
   pool, so ``fq submit`` warns when a job relies on the timeout alone.

2. ``sentinel`` -- opt-in, for guests that cannot exit.
   The runner polls the live uartlog on the run hosts and finishes the job
   when ``sentinel_regex`` appears (or fails it on ``fail_regex``).  Costs an
   ssh round trip per poll, which is why it is not the default.

3. ``timeout`` -- the backstop, always armed.
   Every job has a wall-clock cap.  Under modes ``exit`` and ``sentinel``
   hitting it is a **failure** (state ``TIMEOUT``): the job did not end the
   way it said it would.  Under mode ``timeout`` it is the *intended* end and
   the job is ``DONE`` -- that mode exists for "run this thing for ten minutes
   and give me the uartlog".

A client may also end a job explicitly at any time with ``fq signal-done``,
which is mechanism (4): useful for an external harness that knows when its
work is finished.

Exit codes
----------
The guest's HTIF word carries ``(code << 1) | 1``, so a guest can report
pass/fail.  What ``fq`` records as ``exit_code`` is the exit status of
``firesim runworkload``, which is not the same thing -- the manager does not
plumb the guest's HTIF code out to its own exit status.  A job that needs the
guest's own verdict should assert on it with ``sentinel_regex`` /
``fail_regex``, or parse the collected uartlog.  This is called out here
rather than silently reported as if it were the guest's code.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional

MODES = ("exit", "sentinel", "timeout")


@dataclasses.dataclass
class CompletionPolicy:
    mode: str = "exit"
    sentinel_regex: Optional[str] = None
    fail_regex: Optional[str] = None
    timeout_s: int = 0

    @property
    def watches_uart(self) -> bool:
        """Whether the runner must poll the uartlog while RUN is in flight."""
        return bool(self.sentinel_regex or self.fail_regex)

    @property
    def timeout_is_success(self) -> bool:
        return self.mode == "timeout"

    def compiled(self) -> tuple[Optional[re.Pattern], Optional[re.Pattern]]:
        return (
            re.compile(self.sentinel_regex) if self.sentinel_regex else None,
            re.compile(self.fail_regex) if self.fail_regex else None,
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def from_spec(spec: dict[str, Any], default_timeout_s: int = 0
              ) -> CompletionPolicy:
    """Build a policy from a job spec.

    Accepts both the structured ``completion`` block and the flat
    ``done_regex``/``fail_regex``/``timeout_s`` fields, so simple callers stay
    simple.  The structured block wins where both are given.
    """
    raw = dict(spec.get("completion") or {})
    mode = str(raw.get("mode") or "").strip()
    sentinel = raw.get("sentinel_regex") or spec.get("done_regex")
    fail = raw.get("fail_regex") or spec.get("fail_regex")
    timeout = int(raw.get("timeout_s") or spec.get("timeout_s") or
                  default_timeout_s or 0)
    if not mode:
        # Inferred, not guessed: a sentinel implies sentinel mode.
        mode = "sentinel" if sentinel else "exit"
    return CompletionPolicy(mode=mode, sentinel_regex=sentinel,
                            fail_regex=fail, timeout_s=timeout)


def validate(spec: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    raw = spec.get("completion")
    if raw is not None and not isinstance(raw, dict):
        return ["'completion' must be a mapping"]
    pol = from_spec(spec)
    if pol.mode not in MODES:
        errs.append(f"completion.mode must be one of {'|'.join(MODES)}, "
                    f"got {pol.mode!r}")
    if pol.mode == "sentinel" and not pol.sentinel_regex:
        errs.append("completion.mode='sentinel' needs a sentinel_regex")
    for name, pat in (("sentinel_regex", pol.sentinel_regex),
                      ("fail_regex", pol.fail_regex)):
        if pat:
            try:
                re.compile(pat)
            except re.error as exc:
                errs.append(f"completion.{name} is not a valid regex: {exc}")
    return errs


def advisory(spec: dict[str, Any], timeout_s: int) -> Optional[str]:
    """A warning to show the submitter, or None.

    Surfaced at submit time because by the time it matters the FPGA is
    already being held.
    """
    pol = from_spec(spec, timeout_s)
    if pol.mode == "exit" and not pol.watches_uart:
        return (
            f"job will end when the guest terminates the simulation "
            f"(HTIF exit, e.g. Zephyr sys_reboot(SYS_REBOOT_COLD)). If your "
            f"guest never calls it, this job will hold an FPGA for the full "
            f"{timeout_s}s timeout before being killed. Add "
            f"completion.sentinel_regex if the guest cannot exit on its own.")
    return None
