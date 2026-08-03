# RoSE Flight Controller — Modular Threaded Architecture

Re-architects `samples/rose_flight_controller` from a single `main()` loop into modular task
blocks running as Zephyr threads, supporting **estimation and control at different rates** and
**decoupling IO from compute**. Selected at build time by `-DROSE_THREADED` (default **0** =
single loop; see the caveat below).

## Blocks

```
[sensor IO] --sem_sample--> [estimator] --sem_state--> [control] --sem_done--> back to IO
     |  (reads sensors over the RoSE bridge / real drivers; sends the actuator command)
     +--> g_frame (mutex)        g_state (mutex)          g_ctrl (mutex)
```

- **IO thread** (`io_block`, prio 7): batched sensor fetch (TX) + collect (RX), publishes the
  frame, hands off (`sem_sample`), **blocks on `sem_done`** until compute finishes, then sends
  the fresh actuator command. The blocking handshake keeps the per-grant sequence deterministic
  (the lockstep protocol needs a fixed request/action order per grant).
- **Estimator thread** (`est_block`, prio 5): runs every frame — `est.update` + `get_state`,
  publishes `g_state`, signals `sem_state`.
- **Control thread** (`ctrl_block`, prio 3): runs TinyMPC once every `ROSE_CTRL_DIV` frames →
  **control rate = estimation rate / DIV** (e.g. estimate @200 Hz, control @50 Hz at DIV=4).
  Always signals `sem_done` (on skip grants the last command is held — genuine sub-rate control).
- Blocks communicate via mutex-protected latest-value buffers + binary semaphores. On real
  hardware the IO block's DMA/IRQ transport overlaps with compute; in the single-core lockstep
  co-sim the blocks serialize within each grant, but the structure + rates are the real thing.

Build the threaded architecture with `-DROSE_THREADED=1` (optionally `-DROSE_CTRL_DIV=4`).

## Validation

- **DIV=1 (control = estimation rate):** hovers **identically to the single loop** — z_mean
  1.0232, ripple 0.01 mm, vel ~0 over the steady window. No regression.
- **DIV=4 (control @50 Hz, estimate @200 Hz):** boots "control every 4" and hovers at z=1.023 —
  multi-rate control validated (the 50 Hz-designed TinyMPC at its native rate, estimator faster).

## Known issue — a SEPARATE, systemic co-sim hang (not the threading)

While validating, the co-sim was found to **hang at ~235–236 steps on a perfectly-still clean
hover** — the guest stalls waiting for the synchronizer's next grant while the synchronizer
(Isaac) spins. **This is NOT caused by the threading:** building and running the *committed
single-loop* controller reproduces the identical hang at 236. Moving runs (sensor noise, wind,
maze navigation) sail past it (2400 / 358 / 1600 steps) — i.e. it is correlated with a
perfectly-static hover and appeared after a host reboot, pointing at a physics/PhysX
interaction (e.g. rigid-body sleep) or a synchronizer edge case, independent of this task. A
quick `sleep_threshold=0` attempt via the physx view did not resolve it, so the exact fix is
still open and tracked separately.

**Because of that systemic hang, `ROSE_THREADED` defaults to 0** (the proven single loop) so
the existing stress/navigation flows are unaffected; the threaded blocks are fully implemented
and validated-equivalent, available via `-DROSE_THREADED=1`, and ready to become the default
once the systemic co-sim hang is run down (which will unblock clean-hover runs generally).
