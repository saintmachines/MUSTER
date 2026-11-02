#!/usr/bin/env python3
"""Re-verify MANIFEST_copies.json. Every '# >>> COPY file:a-b dedent=d' block of ship_lme.py and ship_derive.py must have
the recorded sha256, in the recorded order, and each whole file its recorded sha256. With --src DIR (the harness folder)
each block is also compared with the named source lines of the harness, line for line. CPU only. Exit 1 on any difference."""
import argparse, hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="")
A = ap.parse_args()
MAN = json.load(open(os.path.join(HERE, "MANIFEST_copies.json")))
SRCREL = {"lme_pf29.py": "lme_pf29.py", "derive_config.py": "derive_config.py",
          "real_e1_conflict_killer.py": "src/real_e1_conflict_killer.py", "core2.py": "core2.py"}
LINES = {}
if A.src:
    for k, rel in SRCREL.items():
        b = open(os.path.join(A.src, rel), "rb").read()
        assert hashlib.sha256(b).hexdigest() == MAN["sources"][k]["sha256"], ("the harness file changed", k)
        LINES[k] = b.decode("utf-8").split("\n")


def block(f, a, b, ded):
    out = []
    for i in range(a, b + 1):
        ln = LINES[f][i - 1]
        if ded and ln.strip():
            assert ln.startswith(" " * ded)
            ln = ln[ded:]
        out.append(ln)
    return "\n".join(out) + "\n"


bad = 0
for name, rec in MAN["outputs"].items():
    raw = open(os.path.join(HERE, name), "rb").read()
    ok_file = hashlib.sha256(raw).hexdigest() == rec["sha256"]
    got = raw.decode("utf-8").split("\n")
    blocks, i = [], 0
    while i < len(got):
        if got[i].startswith("# >>> COPY "):
            f, rng = got[i][len("# >>> COPY "):].split(" ")[0].split(":")
            ded = int(got[i].split("dedent=")[1])
            a, b = map(int, rng.split("-"))
            j = got.index("# <<< COPY", i)
            body = "\n".join(got[i + 1:j]) + "\n"
            blocks.append((f, a, b, ded, hashlib.sha256(body.encode()).hexdigest(), body))
            i = j
        i += 1
    want = rec["copied_blocks"]
    ok_n = len(blocks) == len(want) == rec["verified_blocks"]
    ok_blocks = ok_n and all(
        (f, [a, b], ded, h) == (w["file"], w["lines"], w["dedent"], w["sha256"]) for (f, a, b, ded, h, _), w in zip(blocks, want))
    ok_src = None
    if A.src:
        ok_src = all(block(f, a, b, ded) == body for (f, a, b, ded, h, body) in blocks)
    lines = sum(b - a + 1 for (f, a, b, ded, h, _) in blocks)
    print("%-15s file sha256 %s | %d of %d blocks equal the manifest (%d copied lines) | against the shipped source lines: %s"
          % (name, "OK" if ok_file else "DIFFERS", sum(1 for (f, a, b, ded, h, _), w in zip(blocks, want) if h == w["sha256"]),
             len(want), lines, {None: "not checked (sources not here)", True: "OK", False: "DIFFERS"}[ok_src]))
    bad += int(not (ok_file and ok_blocks and ok_src is not False))
print("MANIFEST_copies.json: %s" % ("verified" if not bad else "%d file(s) differ" % bad))
sys.exit(1 if bad else 0)
