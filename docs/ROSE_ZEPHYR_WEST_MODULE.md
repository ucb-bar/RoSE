# Making the RoSE Zephyr module standalone via a west manifest project

How the rose Zephyr files are integrated today, why the rose samples in
`zephyr-chipyard-sw` aren't standalone-buildable, and a **validated** fix: register
`zephyr-rose` as a **west manifest project** so Zephyr auto-discovers it — no
`-DZEPHYR_EXTRA_MODULES` injection from the parent RoSE repo.

## Current integration (the problem)

- The `zephyr-rose` **module** (bridge/protocol drivers, virtual sensor shims, `rose/*.h`
  API, and the `ucbbar,rose-*` DT bindings) lives in **RoSE** at `soc/sw/zephyr-rose`
  (`zephyr/module.yml` → `cmake: .`, `kconfig: Kconfig`, `dts_root: .`).
- The **application samples** (`rose_flight_controller`, `rose_nav_controller`,
  `rose_multisensor_probe`, `rose`) live in **zephyr-chipyard-sw** under `samples/`, and
  reference the module (`#include <rose/rose.h>`, `compatible = "ucbbar,rose-imu"`).
- zephyr-chipyard-sw's west workspace has **topdir `zephyr_ws/`** with the **manifest
  being the zephyr fork's own `zephyr/west-riscv.yml`** (a curated RISC-V module set);
  everything else (XNNPACK, executorch, tinympc) comes in as **git submodules**. There is
  **no app-level manifest**, and the rose module is in **neither** the manifest nor a
  submodule of zephyr-chipyard-sw.

The rose samples only build because RoSE's `build_zephyr_rose.sh` injects the module by
absolute path:

```
west build -b spike_riscv64 <sample> -- -DZEPHYR_EXTRA_MODULES=$ROSE_DIR/soc/sw/zephyr-rose
```

Consequently a **standalone** zephyr-chipyard-sw checkout **cannot build the rose samples**:
`ucbbar,rose-imu` is an unknown binding, `<rose/rose.h>` is missing, and the drivers never
compile. (The `-fresh` standalone clone has neither the module nor even the rose samples.)

## The fix: `zephyr-rose` as a west manifest project (validated)

Register the module as a project in the west workspace. Zephyr's build discovers modules by
walking `west list` and scanning each project for `zephyr/module.yml`, so once `zephyr-rose`
is a workspace project it is picked up automatically — `dts_root` registers the
`ucbbar,rose-*` bindings, and the CMake/Kconfig pull in the drivers + subsys.

**Validated offline** (2026-08-03): added `zephyr-rose` as a project to the active manifest,
`west update zephyr-rose` from a local bare repo, then:

```
$ west list | grep zephyr-rose
zephyr-rose  modules/lib/zephyr-rose  main  file:///tmp/rose-mod-remote/zephyr-rose

$ west build -b spike_riscv64 samples/rose_multisensor_probe   # NO -DZEPHYR_EXTRA_MODULES
...
[132/132] Linking CXX executable zephyr/zephyr.elf
BUILD_ELF: PRESENT
```

i.e. the rose sample builds standalone from the workspace, with the module resolved purely
through west. (The `-fresh` workspace was restored afterward.)

## Prerequisite: split `zephyr-rose` into its own repo

A west project is a whole git repo, so `zephyr-rose` must become one — e.g.
**`ucb-bar/zephyr-rose`**. RoSE keeps the *host* half of the protocol (`soc/src/main/cc/
rose_spike`, the synchronizer, and the generated `rose_packet.h`); the *guest* module moves
to its own repo that both RoSE (for co-sim builds) and zephyr-chipyard-sw pull via west.
This preserves protocol cohesion (host + guest still versioned against the same generated
header) while making each consumer's build self-contained.

## Recommended: app-level manifest in zephyr-chipyard-sw

Rather than editing the zephyr fork's `west-riscv.yml` (it shouldn't know about a
chipyard-sw module), give zephyr-chipyard-sw its **own** manifest that imports the fork's
curated manifest and adds `zephyr-rose`:

```yaml
# zephyr-chipyard-sw/west.yml  (app-level manifest)
manifest:
  remotes:
    - name: ucb-bar
      url-base: https://github.com/ucb-bar
  defaults:
    remote: ucb-bar
  projects:
    - name: zephyr
      repo-path: zephyr
      revision: <pinned-rev-or-branch>
      import: west-riscv.yml          # pulls the fork's curated RISC-V module set
    - name: zephyr-rose
      repo-path: zephyr-rose
      revision: main
      path: modules/lib/zephyr-rose   # RoSE bridge module (guest half)
  self:
    path: .
```

Then `west init -l . && west update` assembles the workspace, and `west build` discovers
`zephyr-rose` automatically. (The offline validation above exercised the module-discovery +
build half directly; the `import:` line resolves under a normal networked `west update`,
which establishes the `manifest-rev` ref that west imports require.)

Finally, drop the injection from `soc/sim/build_zephyr_rose.sh`:

```diff
- west build -p always -b spike_riscv64 --build-dir "$OUT/$app" "$src" \
-   -- -DZEPHYR_EXTRA_MODULES="$MODULE"
+ west build -p always -b spike_riscv64 --build-dir "$OUT/$app" "$src"
```

(RoSE would consume `zephyr-rose` the same way — as a west project in its own build
workspace — instead of pointing at `soc/sw/zephyr-rose`.)

## Why not the alternatives

- **ucb-bar/zephyr (the fork):** no — Zephyr modules live outside the tree by design, and
  the rose sensor drivers are **co-sim virtual shims**, not real HW drivers. The real
  drivers the samples bind to on hardware (`st,vl53l5cx`, `bmi088`) already belong upstream;
  the RoSE shims don't.
- **Vendor the module into zephyr-chipyard-sw:** makes it self-contained but splits the
  protocol guest-half from `rose_spike` + the generated `rose_packet.h` in RoSE — the drift
  risk the generated header exists to prevent. Only choose this if standalone buildability
  strictly outranks protocol cohesion.

## Status: split executed (local), publish pending

The history-preserving extraction is **done** (2026-08-04). `git subtree split
--prefix=soc/sw/zephyr-rose` produced a standalone repo with the module's full 5-commit
history (files moved to repo root, `zephyr/module.yml` at top), plus a README commit:

- **Standalone repo:** `../zephyr-rose/` (sibling of the RoSE checkout) — 6 commits, tree
  verified byte-identical to the in-tree `soc/sw/zephyr-rose`.
- **Portable bundle:** `../zephyr-rose.bundle` (32 KB, complete history) — clone-able
  anywhere with `git clone zephyr-rose.bundle`.

The two remaining steps need the live remote (there is no `gh` on this host, so the repo
must be created on GitHub first):

1. **Create** an *empty* `ucb-bar/zephyr-rose` on GitHub (no README/license, so the pushed
   history is the only content).
2. **Publish + swap** — run the helper, which pushes the standalone repo and converts RoSE's
   in-tree directory into a submodule:

   ```
   bash soc/sim/finalize_zephyr_rose_split.sh
   # or target a fork first:  REMOTE=git@github.com:<you>/zephyr-rose.git bash soc/sim/finalize_zephyr_rose_split.sh
   ```

   The RoSE build is **unaffected**: `build_zephyr_rose.sh` injects the module by *path*
   (`-DZEPHYR_EXTRA_MODULES=$ROSE_DIR/soc/sw/zephyr-rose`), and the submodule checkout sits
   at that same path — no build-script change required. (Moving RoSE itself onto the
   west-project consumption path, and dropping the `-DZEPHYR_EXTRA_MODULES` flag, stays an
   optional later cleanup.)

3. **zephyr-chipyard-sw:** add the app-level `west.yml` above, `west update`, then its rose
   samples build standalone (no `ZEPHYR_EXTRA_MODULES`).
4. (Bonus) the low-level bridge validators (`rxvalidate`/`protovalidate`/`dmavalidate`)
   ride along as module samples inside `zephyr-rose`; the application samples stay in
   zephyr-chipyard-sw with their estimator/TinyMPC deps.
