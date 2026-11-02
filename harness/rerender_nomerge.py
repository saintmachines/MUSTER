"""Re-render a MUSTER export with every unit as its own block (no merging), 2026-09-28.
Units are rebuilt with the harness chunker (ATOM=blank, CHUNK=512, o200k as in the harness; lme_pf29.py:1219) and parents with
MSCALE labelling (lme_pf29.py:2198). Each exported block is decomposed into unit ids; the decomposition must
re-join to the exported context byte for byte, or the question is refused. The no-merge render is the harness
branch for len(fed) <= lim (lme_pf29.py:357-360): "Memory n:" per unit, units in id order.
Usage: [ATOM=blank|sentence CHUNK=512 MSCALE=4096] python rerender_nomerge.py <parquet> <in_shard_prefix> <out_prefix>"""
import sys, re, json, glob, hashlib, tiktoken, pandas as pd
ENC = tiktoken.encoding_for_model("gpt-4o-mini")   # the harness tokenizer, lme_pf29.py:248


import os
ATOM = os.environ.get("ATOM", "blank"); CS = int(os.environ.get("CHUNK", "512")); MSC = int(os.environ.get("MSCALE", "4096"))
JN = {"blank": "\n\n", "sentence": " "}[ATOM]


def split_atoms(text):
    if ATOM == "blank":
        return [x for x in re.split(r"\n\s*\n", text) if x.strip()]
    import nltk
    return nltk.sent_tokenize(text)   # lme_pf29._atom_split("sentence")


def chunk(text, cs=None):
    cs = cs or CS
    atoms = split_atoms(text)
    out, cur, n = [], [], 0
    for a in atoms:
        t = ENC.encode(a, allowed_special={"<|endoftext|>"})
        if len(t) > cs:
            if cur:
                out.append(JN.join(cur)); cur, n = [], 0
            for o in range(0, len(t), cs):
                out.append(ENC.decode(t[o:o + cs]))
            continue
        if n + len(t) > cs and cur:
            out.append(JN.join(cur)); cur, n = [], 0
        cur.append(a); n += len(t)
    if cur:
        out.append(JN.join(cur))
    return out


pq, inpre, outpre = sys.argv[1:4]
df = pd.read_parquet(pq)
CH = {}


def units(row):
    if row not in CH:
        ch = chunk(str(df["context"].iloc[row])); par, cum = [], 0
        for s in ch:
            par.append(cum // MSC); cum += len(ENC.encode(s))
        idx = {}
        for i, s in enumerate(ch):
            idx.setdefault(s[:80], []).append(i)
        CH[row] = (ch, par, idx)
    return CH[row]


def decompose(block, ch, idx):
    ids, p = [], 0
    while p < len(block):
        cand = [i for i in idx.get(block[p:p + 80], []) if block.startswith(ch[i], p)
                and (p + len(ch[i]) == len(block) or block[p + len(ch[i])] == "\n")]
        if not cand:
            return None
        i = max(cand, key=lambda j: len(ch[j])); ids.append(i); p += len(ch[i])
        if p < len(block):
            p += 1
    return ids


HDR = re.compile(r"(?m)^Memory \d+:\n")
stat = dict(q=0, ok=0, bad=0, blocks=0, xpar=0, units=0)
for f in sorted(glob.glob(inpre + "_s*.jsonl")):
    out = open(outpre + f[len(inpre):], "w")
    for l in open(f):
        r = json.loads(l); stat["q"] += 1
        ch, par, idx = units(r["row"])
        new = {}
        for k in [k for k in r if k.startswith("ctx|")]:
            c = r[k]; heads = HDR.findall(c); parts = HDR.split(c)
            if parts[0] != "":
                new = None; break
            body = parts[1:]
            blocks = [b[:-1] if (n < len(body) - 1 and b.endswith("\n")) else b for n, b in enumerate(body)]
            dec = [decompose(b, ch, idx) for b in blocks]
            if any(d is None for d in dec):
                new = None; break
            back = "\n".join(h + "\n".join(ch[i] for i in d) for h, d in zip(heads, dec))
            if back != c:
                new = None; break
            fed = [i for d in dec for i in d]
            stat["blocks"] += len(dec); stat["xpar"] += sum(len({par[i] for i in d}) > 1 for d in dec)
            stat["units"] += len(fed)
            new[k] = "\n".join("Memory %d:\n%s" % (n + 1, ch[i]) for n, i in enumerate(sorted(fed)))
        if new is None:
            stat["bad"] += 1; continue
        stat["ok"] += 1
        for k, v in new.items():
            r[k] = v; r["ctxhash|" + k[4:]] = hashlib.sha1(v.encode()).hexdigest()[:12]
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(stat))
