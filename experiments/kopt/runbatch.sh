#!/usr/bin/env bash
source /scratch/dima/rose-infra/RoSE/experiments/kopt/env.sh
exec python /scratch/dima/rose-infra/RoSE/experiments/kopt/fqbatch.py "$@"
