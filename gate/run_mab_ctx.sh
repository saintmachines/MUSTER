#!/bin/bash
# Gate stage 0d: the render chains of the paper's final MemoryAgentBench board and of LoCoMo / LoCoMo-Refined at ten units.
# From the published context dumps (DRY=1 DUMP_CTX=1) of the twelve MemoryAgentBench retrieval legs in harness/len_legs.json
# (lz_<cell>) and of the two FactConsolidation structural legs (mg_fc{sh,mh}_struct), rebuild with the released code every
# bundle the board read: make_mab_bundle.py on all fourteen cells; render_rule.py R4 (one block per parent, stored order;
# ATOM, CHUNK and MSCALE of the leg) on EventQA, DetectiveQA and InfBench-Sum; "[CURRENT] " removed on FC-SH and FC-MH. Then
# LoCoMo and LoCoMo-Refined at U=10: make_bundle.py, then render_rule.py R4 with ATOM=blank CHUNK=512 MSCALE=2048.
# Reads the experiment tree (EXPERIMENT_DIR), writes only under GATE_DIR/mab_ctx. CPU only: no model, no API call.
#   GATE_DIR=... EXPERIMENT_DIR=... [HARNESS_DIR=$GATE_DIR/harness] [PYTHON=python3] bash run_mab_ctx.sh
# then: python compare_mab_ctx.py (same environment) prints the comparison against the board's own records.
set -u
G=${GATE_DIR:?set GATE_DIR to the gate directory (writable)}; M=${EXPERIMENT_DIR:?set EXPERIMENT_DIR to the read-only experiment tree}
H=${HARNESS_DIR:-$G/harness}; PY=${PYTHON:-python3}; W=$G/mab_ctx
mkdir -p $W/results $W/bundle_out
ln -sfn $M/data $W/data; ln -sfn $M/data_all $W/data_all
export PYTHONHASHSEED=0
leg () {  # arm ds parquet rows nq ATOM CHUNK MSCALE of leg lz_<cell>, from the released leg table
  $PY -c "import json; l=[x for x in json.load(open('$H/len_legs.json'))['legs'] if x['tag']=='lz_$1'][0]; e=l['env']
print(l['arm'].split('|')[0], l['ds'], l['parq'], l['rows'].strip(','), l['nq'], e['ATOM'], e['CHUNK'], e['MSCALE'])"; }
echo "[$(date +%H:%M:%S)] MemoryAgentBench, twelve retrieval legs"
for C in shqa mhqa lme rec evq det summ b77 clin nlu trcC trcF; do
  read ARM DS PQ RW NQ ATOM CHUNK MSCALE <<< "$(leg $C)"
  ln -sfn $M/len0928_in/lz_${C}_s0.jsonl $W/results/lz_$C.jsonl
  ( cd $W && DSNAME=$DS MUSTER_HOME=$W $PY $H/make_mab_bundle.py lz_$C $ARM $PQ $RW $NQ - $W/bundle_out/full_lz_$C.jsonl | grep -v '^BUNDLE_JSON' ) || echo "FAIL bundle $C"
  case $C in evq|det|summ)
    ATOM=$ATOM CHUNK=$CHUNK MSCALE=$MSCALE $PY $H/render_rule.py R4 $W/$PQ $W/bundle_out/full_lz_$C.jsonl $W/bundle_out/r4_lz_$C.jsonl || echo "FAIL R4 $C";;
  esac
done
echo "[$(date +%H:%M:%S)] MemoryAgentBench, FactConsolidation"
CR=data_all/data/Conflict_Resolution-00000-of-00001.parquet
for spec in "fcsh 7" "fcmh 3"; do
  read C RW <<< "$spec"
  ln -sfn $M/results/mg_${C}_struct.jsonl $W/results/mg_${C}_struct.jsonl
  ( cd $W && MUSTER_HOME=$W $PY $H/make_mab_bundle.py mg_${C}_struct attprio_sess $CR $RW 100 - $W/bundle_out/mg_${C}_struct.jsonl | grep -v '^BUNDLE_JSON' ) || echo "FAIL bundle $C"
  $PY - $W/bundle_out/mg_${C}_struct.jsonl $W/bundle_out/mg_${C}_struct_notag.jsonl <<'PYEOF'
import json, hashlib, sys
src, dst = sys.argv[1:3]; rows = [json.loads(l) for l in open(src)]; n_tag = 0
for d in rows:
    n_tag += d['context'].count('[CURRENT] ')
    d['context'] = d['context'].replace('[CURRENT] ', ''); assert '[CURRENT]' not in d['context']
    d['ctxhash'] = hashlib.sha1(d['context'].encode()).hexdigest()[:12]
with open(dst, 'w') as fo:
    for d in rows: fo.write(json.dumps(d, ensure_ascii=False) + '\n')
print('NOTAG', dst, len(rows), 'rows,', n_tag, '"[CURRENT] " removed')
PYEOF
done
echo "[$(date +%H:%M:%S)] LoCoMo and LoCoMo-Refined, U=10"
for spec in "lcm_u10 locomo.parquet" "lcmrefc_u10 locomo_refined_conv.parquet"; do
  read T PQ <<< "$spec"
  ( cd $H && MUSTER_HOME=$H MUSTER_RESULTS=$M/results $PY make_bundle.py $T $M/beam/$PQ $W/bundle_out/${T}_r0.jsonl | tail -1 ) || echo "FAIL bundle $T"
  ATOM=blank CHUNK=512 MSCALE=2048 $PY $H/render_rule.py R4 $M/beam/$PQ $W/bundle_out/${T}_r0.jsonl $W/bundle_out/${T}_r4.jsonl || echo "FAIL R4 $T"
done
echo "[$(date +%H:%M:%S)] STAGE 0d BUILD DONE"
