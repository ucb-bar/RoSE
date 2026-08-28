#!/usr/bin/env bash
# Run the whole fq test suite. No AWS, no FireSim, no FPGA required.
#   ./run_tests.sh            # everything
#   ./run_tests.sh -f         # fast: unit tests only (no daemon, ~0.1s)
#   ./run_tests.sh tests.test_scheduler   # one module
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::ResourceWarning}"

if [[ "${1:-}" == "-f" || "${1:-}" == "--fast" ]]; then
  exec python3 -m unittest -v \
    tests.test_scheduler tests.test_config tests.test_locks \
    tests.test_completion tests.test_occupancy
fi
if [[ $# -gt 0 ]]; then
  exec python3 -m unittest -v "$@"
fi
exec python3 -m unittest discover -s tests -t . -p 'test_*.py' -v
