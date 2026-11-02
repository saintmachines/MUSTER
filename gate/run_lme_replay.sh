#!/bin/bash
# MemoryAgentBench LongMemEval cell: CPU-only replay of the published board leg p24_lme_m (env `BLME` of the experiment
# machine's mabglm_legs.sh, reproduced verbatim below) with DRY=1 DRY_XC=1 DUMP_CTX=1: retrieval, cached extraction
# (587 extraction-cache hits on the published replay, 0 API calls), annotation and render; no model call. The text oracle
# is results/mg_lme_on.jsonl (300 questions, ctxhash equal to p24_lme_m on 300/300). Writes only under the gate directory.
#   GATE_DIR=... EXPERIMENT_DIR=... DRY=1 bash run_lme_replay.sh
set -u
G=${GATE_DIR:?set GATE_DIR to the gate directory (writable)}; M=${EXPERIMENT_DIR:?set EXPERIMENT_DIR to the read-only experiment tree}; PY=${PYTHON:-python3}
cd $G/harness
BLME='PYTHONHASHSEED=0 PER_ARM=1 BM25LC=1 CEML=1024 CEBS=16 CE_SUBMAX=0 MR_ALPHA=0 DOCFLOOR=0 DOCU=0 NOREFILL=0 XQUOTAS=4,3,3 METHODS=flrs_4_3_3 LPROFILE=0 PROFILE=0 U=10 ATOM=sentence CHUNK=512 HDETECT=0 MSCALE=4096 K1=200'
nohup env $BLME OAMODEL=gpt-4o-mini-2024-07-18 DRY=1 DRY_XC=1 DUMP_CTX=1 RESUME=0 BGE_DEVICE=cpu CE_DEVICE=cpu HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  CE_CACHE=$G/ce_cache XC_DIR=$G/extract_cache COST_LEDGER_DIR=$G/cost_ledger MUSTER_HOME=$G/harness \
  PARQ=$M/data/Accurate_Retrieval-00000-of-00001.parquet ROWS=17,18,19,20,21 NQ=300 CAP_USD=0.01 GUARD=gate_lme \
  OUT=$G/results/gate_lme.jsonl nice -n 10 $PY lme_pf29.py > $G/logs/gate_lme.log 2>&1 &
P=$!; echo "lme replay pid $P"
cat > $G/logs/watch_lme.sh <<W
P=$P; DL=\$(( \$(date +%s) + 7200 )); MAX=0
while kill -0 \$P 2>/dev/null; do r=\$(ps -o rss= -p \$P | tr -d ' '); [ -n "\$r" ] && [ \$r -gt \$MAX ] && MAX=\$r
  [ \$(date +%s) -gt \$DL ] && { kill \$P; echo "FAIL deadline"; exit 3; }
  [ \${r:-0} -gt \$(( 40*1024*1024 )) ] && { kill \$P; echo "FAIL rss \$r KB"; exit 4; }
  sleep 15; done; echo "END maxrss_gib=\$(( MAX/1024/1024 )) rows=\$(wc -l < $G/results/gate_lme.jsonl 2>/dev/null)"
W
nohup bash $G/logs/watch_lme.sh > $G/logs/watch_lme.log 2>&1 &
