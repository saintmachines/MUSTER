"""Legacy four-stage build-time probe, retained for the historical measurement and control.

This probe uses cl100k_base, units of up to 512 tokens, bge-m3 at max_seq 448,
batch 32, and BM25 over lowercased space-split words. It excludes the parent graph.
Its four stages are statistics, units, one embedding pass, and BM25.

The paper's current build-time measurements use o200k_base, max_seq 2048,
and include the parent graph. See ../gate/build_time_rtx.md for the measured
RTX PRO 6000 fp16 runs and their settings. The driver for those exact runs is
not included in this directory; running this legacy probe measures a different
configuration. No generative model is called.

Run on a clean machine (Colab RTX PRO 6000 or an embedding API).
Usage:
  GPU (Colab):  pip install sentence-transformers rank_bm25 tiktoken ; python time_build.py --device cuda
  API:          OPENAI_API_KEY=... python time_build.py --api text-embedding-3-small --workers 8
Inputs: beam_{100K,1M,10M}_conv0.txt next to this file (conversation 0 of each scale, the benchmark's context string)."""
import argparse, time, os, re, json, concurrent.futures as cf
import tiktoken
from rank_bm25 import BM25Okapi
ap = argparse.ArgumentParser(); ap.add_argument('--device', default=None); ap.add_argument('--api', default=None)
ap.add_argument('--workers', type=int, default=8); ap.add_argument('--scales', default='100K,1M,10M'); a = ap.parse_args()
enc = tiktoken.get_encoding('cl100k_base')
def units_of(text, cap=512):
    atoms = [x for x in re.split(r'\n\s*\n', text) if x.strip()]
    out, cur, n = [], [], 0
    for at in atoms:
        t = len(enc.encode(at, disallowed_special=()))
        if t > cap:
            if cur: out.append('\n\n'.join(cur)); cur, n = [], 0
            ids = enc.encode(at, disallowed_special=()); out += [enc.decode(ids[i:i + cap]) for i in range(0, len(ids), cap)]
        elif n + t > cap:
            out.append('\n\n'.join(cur)); cur, n = [at], t
        else:
            cur.append(at); n += t
    if cur: out.append('\n\n'.join(cur))
    return out
model = None
if a.device:
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time(); model = SentenceTransformer('BAAI/bge-m3', device=a.device); model.max_seq_length = 448
    if a.device == 'cuda': model.half()
    load_s = time.time() - t0
    print(json.dumps({'device': a.device, 'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else a.device, 'model_load_s': round(load_s, 1)}))
    model.encode(['warm up'] * 64, batch_size=32)
else:
    from openai import OpenAI
    cli = OpenAI()
    def embed_batch(b):
        for k in range(6):
            try: return cli.embeddings.create(model=a.api, input=b).data
            except Exception:
                time.sleep(2 ** k)
        raise RuntimeError('embedding failed')
for sz in a.scales.split(','):
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'beam_{sz}_conv0.txt')).read()
    T = {}; t = time.time()
    src_tok = len(enc.encode(text, disallowed_special=()))
    lines = text.split('\n'); serial = sum(1 for l in lines if re.match(r'^\s*(\d+)\.\s+(.+)$', l)) / max(1, len(lines))
    T['statistics_s'] = time.time() - t; t = time.time()
    U = units_of(text); T['units_s'] = time.time() - t; t = time.time()
    if model is not None:
        model.encode(U, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    else:
        bs = [U[i:i + 256] for i in range(0, len(U), 256)]
        with cf.ThreadPoolExecutor(a.workers) as ex: list(ex.map(embed_batch, bs))
    T['embed_s'] = time.time() - t; t = time.time()
    BM25Okapi([u.lower().split() for u in U]); T['bm25_s'] = time.time() - t
    T = {k: round(v, 2) for k, v in T.items()}
    print(json.dumps({'scale': sz, 'source_tokens': src_tok, 'units': len(U), 'serial_frac': round(serial, 3), **T,
                      'build_total_s': round(sum(T.values()), 1), 'embedder': a.api or f'bge-m3 ({a.device})'}), flush=True)
