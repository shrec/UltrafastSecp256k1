#!/usr/bin/env bash
# ===========================================================================
# Final unified benchmark for the CPU representation search.
#
# Run this on a QUIET machine. Everything the search produced is measured here
# in one session, against one baseline, under the repository's canonical
# methodology -- so the numbers are comparable to docs/BITCOIN_CORE_BENCH_RESULTS.json
# rather than to each other.
#
#   sudo ./tools/run_final_benchmark.sh          # locks frequency, runs everything
#   ./tools/run_final_benchmark.sh --no-lock     # skips the sudo steps, noisier
#
# WHY THE FREQUENCY LOCK MATTERS HERE. Every inconclusive result in this search
# came from frequency variation, not from the change under test. The canonical
# methodology in BITCOIN_CORE_BENCH_RESULTS.json is
#     cpupower frequency-set -g performance + intel_pstate/no_turbo=1,
#     taskset -c 0, nice -20, 5 runs, bench_bitcoin -min-time=3000
# and the published +1.2% ConnectBlock margin was measured that way. A result
# produced any other way cannot be compared to it.
# ===========================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="out/representation-search-cpu/final"
mkdir -p "$OUT"

LOCK=1
[[ "${1:-}" == "--no-lock" ]] && LOCK=0

# --- frequency lock ------------------------------------------------------
if [[ $LOCK -eq 1 ]]; then
  echo "== locking CPU frequency (canonical methodology) =="
  cpupower frequency-set -g performance >/dev/null 2>&1 \
    && echo "   governor  -> performance" || echo "   governor  -> FAILED (need sudo)"
  if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo && echo "   turbo     -> disabled"
  else
    echo "   turbo     -> FAILED (need sudo); results will carry turbo variance"
  fi
else
  echo "== --no-lock: frequency NOT pinned, treat every margin under 3% as noise =="
fi
echo "   governor now: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "   load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo

# Core 0 is a P-core on this machine (siblings 0-1, 4.7 GHz; cores 12-15 are
# E-cores at 3.5 GHz and must never be used for timing).
CORE=0
RUN="taskset -c $CORE nice -n -20"

# --- the builds under test ----------------------------------------------
# Every one is the SAME source tree, differing only in -D flags, so the tree is
# never a confounder.
declare -A BUILDS=(
  [baseline]=""
  [coz]="-DREPSEARCH_COZ_TABLE=1"
  [inv]="-DREPSEARCH_CT_SAFEGCD_INV=1"
  [all]="-DREPSEARCH_COZ_TABLE=1 -DREPSEARCH_CT_SAFEGCD_INV=1"
  [all_w13]="-DREPSEARCH_COZ_TABLE=1 -DREPSEARCH_CT_SAFEGCD_INV=1 -DREPSEARCH_DUALMUL_WINDOW_G=13"
  [all_w12]="-DREPSEARCH_COZ_TABLE=1 -DREPSEARCH_CT_SAFEGCD_INV=1 -DREPSEARCH_DUALMUL_WINDOW_G=12"
)

echo "== building ${#BUILDS[@]} variants =="
for name in "${!BUILDS[@]}"; do
  dir="out/final-$name"
  flags="${BUILDS[$name]}"
  if [[ -x "$dir/src/cpu/bench_unified" ]]; then
    echo "   $name: already built"
    continue
  fi
  printf "   %-10s configuring+building..." "$name"
  cmake --preset cpu-release -B "$dir" ${flags:+-DCMAKE_CXX_FLAGS="$flags"} \
      > "$OUT/build-$name.log" 2>&1 \
    && cmake --build "$dir" -j"$(nproc)" --target bench_unified unified_audit_runner \
      >> "$OUT/build-$name.log" 2>&1 \
    && echo " OK" || { echo " FAILED (see $OUT/build-$name.log)"; }
done
echo

# --- correctness gate: no timing until every variant is clean ------------
echo "== CaaS audit on every variant (correctness gates the measurement) =="
audit_fail=0
for name in "${!BUILDS[@]}"; do
  bin="out/final-$name/audit/unified_audit_runner"
  [[ -x "$bin" ]] || { echo "   $name: NO BINARY"; audit_fail=1; continue; }
  ( cd "out/final-$name" && ./audit/unified_audit_runner \
      --json "../../$OUT/audit-$name.json" > "../../$OUT/audit-$name.log" 2>&1 )
  line=$(grep -E 'TOTAL: [0-9]+/[0-9]+' "$OUT/audit-$name.log" | tail -1)
  echo "   $name: $line"
  grep -q 'ALL PASSED' "$OUT/audit-$name.log" || { echo "      ^ NOT all-passed"; audit_fail=1; }
done
if [[ $audit_fail -ne 0 ]]; then
  echo
  echo "AUDIT NOT CLEAN -- refusing to report timings. Fix correctness first."
  exit 1
fi
echo

# --- bench_unified, interleaved, warm-up discarded -----------------------
echo "== bench_unified: 1 discarded warm-up + 5 rounds, rotating order =="
names=("${!BUILDS[@]}")
for name in "${names[@]}"; do
  $RUN "out/final-$name/src/cpu/bench_unified" --json "$OUT/warm-$name.json" >/dev/null 2>&1
done
for r in 0 1 2 3 4; do
  for ((i=0; i<${#names[@]}; i++)); do
    k=$(( (i + r) % ${#names[@]} ))          # rotate so none sits in a cold slot
    n="${names[$k]}"
    $RUN "out/final-$n/src/cpu/bench_unified" --json "$OUT/r${r}-$n.json" >/dev/null 2>&1
    echo "   round $r  $n"
  done
done
echo

python3 experiments/representation_search/tools/compare_final.py "$OUT" | tee "$OUT/SUMMARY.txt"

cat <<'NOTE'

== NEXT: Bitcoin Core, the metric that actually decides ==
bench_unified measures primitives in isolation, which systematically FLATTERS a
large precomputation table -- nothing else competes for L2. ConnectBlock does
not, and it is where this engine's margin over libsecp is thinnest (+1.2%).

  cd ../../bitcoin-core-dev/src/ultrafast_secp256k1
  git fetch origin experiment/representation-search
  git checkout experiment/representation-search
  cd ../..
  cmake -B build-ultra-coz -DSECP256K1_BACKEND=ultrafast \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DCMAKE_CXX_FLAGS="-DREPSEARCH_COZ_TABLE=1 -DREPSEARCH_CT_SAFEGCD_INV=1"
  cmake --build build-ultra-coz -j$(nproc) --target bench_bitcoin

  for b in build-ultra-coz build-ultra-lto build-libsecp-lto; do
    taskset -c 0 nice -20 $b/bin/bench_bitcoin \
      -filter='ConnectBlock.*|SignTransaction.*|SignSchnorr.*' -min-time=3000
  done

Compare against docs/BITCOIN_CORE_BENCH_RESULTS.json, which was produced with
exactly this methodology on this machine.
NOTE
