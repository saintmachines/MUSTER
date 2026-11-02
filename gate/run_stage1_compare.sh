#!/bin/bash
# Stage 1 comparison on the gate machine: a CPU replay (harness OUT with DUMP_CTX, one file per leg) against the published
# R1 bundle, question by question; then the replay is packed as a bundle and re-rendered with R4 and both files are hashed
# against the oracles. Reads the experiment tree (EXPERIMENT_DIR), writes only under the gate directory (GATE_DIR).
#   bash run_stage1_compare.sh <U> <SZ>      e.g. bash run_stage1_compare.sh 20 100k
set -u
S=$(cd "$(dirname "$0")" && pwd)   # this gate directory, where gate_compare.py ships
U=$1; SZ=$2
G=${GATE_DIR:?set GATE_DIR to the gate directory (writable)}; M=${EXPERIMENT_DIR:?set EXPERIMENT_DIR to the read-only experiment tree}; PY=${PYTHON:-python3}
TAG=gate_u${U}_${SZ}; OUT=$G/results/$TAG.jsonl
R1=$M/bundle_out/len0928/full_lz_u${U}_beam${SZ}.jsonl; R4=$M/bundle_out/rules/r4_u${U}_${SZ}.jsonl
cd $G/harness; export PYTHONHASHSEED=0 MUSTER_HOME=$G/harness MUSTER_RESULTS=$G/results
echo "== per-question comparison: $OUT vs $R1 =="
$PY $S/gate_compare.py $OUT $R1
echo "== bundle + R4 re-render from the replay =="
[ -e $G/results/${TAG}_s0.jsonl ] || ln -s $OUT $G/results/${TAG}_s0.jsonl
$PY make_bundle.py $TAG $M/beam/beam${SZ}.parquet $G/bundle_out/replay_r1_u${U}_${SZ}.jsonl | tail -1
ATOM=blank CHUNK=512 MSCALE=4096 $PY render_rule.py R4 $M/beam/beam${SZ}.parquet $G/bundle_out/replay_r1_u${U}_${SZ}.jsonl $G/bundle_out/replay_r4_u${U}_${SZ}.jsonl | tail -1
a=$(shasum -a 256 $G/bundle_out/replay_r1_u${U}_${SZ}.jsonl | cut -c1-12); b=$(shasum -a 256 $R1 | cut -c1-12)
c=$(shasum -a 256 $G/bundle_out/replay_r4_u${U}_${SZ}.jsonl | cut -c1-12); d=$(shasum -a 256 $R4 | cut -c1-12)
echo "RESULT $SZ U=$U  R1 replay=$a oracle=$b $([ $a = $b ] && echo MATCH || echo DIFF)   R4 replay=$c oracle=$d $([ $c = $d ] && echo MATCH || echo DIFF)"
echo "== cache integrity (a changed or new file = a cross-encoder miss during the replay) =="
cd $G/ce_cache && shasum -a 256 -c $G/ce_cache.SHA256 2>/dev/null | grep -v ": OK" | head -5; echo "cache files now: $(ls | wc -l) (snapshot had $(wc -l < $G/ce_cache.SHA256))"
