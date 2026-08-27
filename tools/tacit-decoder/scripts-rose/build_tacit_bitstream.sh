#!/usr/bin/env bash
# Build the Saturn+RoSE+TACIT bitstream (RoseTLRocketSaturnTacitMMIOOnlyConfig): RoSE co-sim
# bridge + the ported TACIT streaming bridge. Needs tacit@rose-tacit-bridge +
# rocket-chip@rose-tacit-encoder checked out (already are). Elaboration first (fail-fast),
# then Vivado (~2.5h). Run AFTER the DMA-instrumented build (one build host).
set -uo pipefail
FSIMDIR=/scratch/dima/rose-infra/RoSE/soc/sim/chipyard/sims/firesim
RECIPE=alveo_u250_firesim-rocket-saturn-tacit-with-rose-fast-no-nic-l2-llc4mb-ddr3
echo "=== TACIT BUILD START $(date) ==="
( set +u
  cd "$FSIMDIR" && source env.sh >/dev/null 2>&1 && source sourceme-manager.sh --skip-ssh-setup >/dev/null 2>&1
  cd deploy
  python3 - "$RECIPE" <<'PY'
import sys, yaml
c = yaml.safe_load(open('config_build.yaml'))
c['builds_to_run'] = [sys.argv[1]]
yaml.safe_dump(c, open('config_build_tacit.yaml','w'), sort_keys=False)
PY
  firesim buildbitstream -b config_build_tacit.yaml -r config_build_recipes.yaml )
echo "=== TACIT BUILD EXIT=$? $(date) ==="
