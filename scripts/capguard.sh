#!/bin/bash
#SBATCH --job-name=aar-capguard
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --output=/opt/aar/work/exp/logs/capguard_%j.out
#
# BACKSTOP: guarantee every AAR chain STOPS at 100 iterations even if the in-process
# MAX_ITERATIONS cap fails. Each loop, for every running aar chain, find the highest
# [Session N] in its CURRENT .out (per-process count, matching the harness cap) and
# scancel it once N >= 100. Eval workers self-exit on idle; dashboards are LEFT UP so
# you can still view final results. gpu:0, so it costs nothing.
set -uo pipefail
LOGS=/opt/aar/work/exp/logs
CAP="${1:-100}"
echo "[capguard] watching aar chains; scancel at session >= ${CAP}"
while true; do
  while IFS='|' read -r jid jn; do
    [ -n "${jid:-}" ] || continue
    out=$(ls -t "${LOGS}/${jn}_${jid}.out" 2>/dev/null | head -1)
    [ -f "$out" ] || continue
    n=$(grep -oE '\[Session [0-9]+\]' "$out" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
    [ -n "${n:-}" ] || continue
    if [ "$n" -ge "$CAP" ] 2>/dev/null; then
      echo "[capguard] $(date '+%F %T') ${jn} (${jid}) hit session ${n} >= ${CAP} -> scancel"
      scancel "$jid"
    fi
  done < <(squeue -u aar-user -h -o '%i|%j' 2>/dev/null | grep -E '\|aar-(refusal|sycophancy|honesty)-')
  sleep 300
done
