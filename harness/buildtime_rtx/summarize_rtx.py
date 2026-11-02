#!/usr/bin/env python3
"""Summarise RESULTS_DIR/beam_buildtime_rtx_repeat*.json into RESULTS_DIR/summary.md and summary.json. CPU only, no torch.
Only the repeats of one setup are summarised: the setup of the running measurement, or of the newest repeat file when
this is run by hand (the same expected.json, stores, device, code sha256, library versions and GPU). Repeats saved with
another setup are named and left out. First the total of the stores: repeat 1 on its own (the first repeat after the
model load and warm-up in its process), then the mean and every repeat, each also with the graph stage. Then per process
the model load and warm-up (each on its own line), the GPU, driver and versions. Then per store: units, parents, and for
each stage and the build total the mean and every repeat's value, and repeat 1's build total. Then the two shipped
assertions inside Units s (re-cut and non-space check) and the fp32 embed pass, which has its own file of the same setup
and is never added to the build totals.
Usage: python -B summarize_rtx.py RESULTS_DIR"""
import glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beambt_common as K

STAGES = (("statistics_s", "Statistics s"), ("units_s", "Units s"), ("embed_s", "Embed s"), ("bm25_s", "BM25 s"),
          ("total_s", "Build total s"), ("graph_s", "Graph s (apart)"), ("total_with_graph_s", "Build with graph s"))
EXTRA = ("units_recut_s", "units_nonspace_s")
FP32_FILE = "beam_buildtime_rtx_fp32.json"


def load(out):
    reps = []
    for p in sorted(glob.glob(os.path.join(out, "beam_buildtime_rtx_repeat*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("complete") is True:
            d["_file"] = os.path.basename(p)
            reps.append(d)
    reps.sort(key=lambda d: d["run"])
    return reps


def load_extra(out, name, setup):                   # a complete fp32 file of the same setup, or None
    try:
        d = json.load(open(os.path.join(out, name)))
    except Exception:
        return None
    return d if (d.get("complete") is True and d.get("setup") == setup) else None


def cell(vals):
    return "%.3f [%s]" % (sum(vals) / len(vals), ", ".join("%.3f" % v for v in vals))


def write(out, want=3, setup=None):
    allr = load(out)
    if not allr:
        return None
    if setup is None:                               # by hand: the setup of the newest complete repeat
        setup = max(allr, key=lambda d: d.get("finished", "")).get("setup")
    reps = [r for r in allr if r.get("setup") == setup]
    left_out = [r["_file"] for r in allr if r.get("setup") != setup]
    if not reps:
        return None
    cells = [c for c in K.ORDER if all(c in r["stores"] for r in reps)]
    test = any(r["meta"].get("emb_limit") for r in reps)
    r1 = next((r for r in reps if r["run"] == 1), None)
    J = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repeats_complete": [r["run"] for r in reps], "repeats_wanted": want,
         "files": [r["_file"] for r in reps], "left_out_other_setup": left_out, "setup": setup, "test_only": test,
         "first_repeat_in_process": {str(r["run"]): bool(r.get("first_repeat_in_process")) for r in reps},
         "stores": {}, "all_stores": {}}
    for c in cells:
        S = [r["stores"][c] for r in reps]
        for k in ("units", "parents", "texts_embedded", "units_sha256", "emb_maxlen"):
            assert all(s[k] == S[0][k] for s in S), (c, k)
        J["stores"][c] = {"name": K.NAME[c], "units": S[0]["units"], "parents": S[0]["parents"],
                          "texts_embedded": S[0]["texts_embedded"], "emb_maxlen": S[0]["emb_maxlen"],
                          "units_sha256": S[0]["units_sha256"]}
        for k in [s[0] for s in STAGES] + list(EXTRA):
            J["stores"][c][k] = {"mean": sum(s[k] for s in S) / len(S), "values": [s[k] for s in S],
                                 "repeat1": r1["stores"][c][k] if r1 else None}
    for k in [s[0] for s in STAGES] + list(EXTRA):
        vals = [sum(r["stores"][c][k] for c in cells) for r in reps]
        J["all_stores"][k] = {"mean": sum(vals) / len(vals), "values": vals,
                              "repeat1": sum(r1["stores"][c][k] for c in cells) if r1 else None}
    J["all_stores"]["units"] = sum(J["stores"][c]["units"] for c in cells)
    J["all_stores"]["parents"] = sum(J["stores"][c]["parents"] for c in cells)
    J["all_stores"]["n_stores"] = len(cells)
    sess = {}
    for r in reps:
        m = r["meta"]
        cu = m.get("cuda") or {}
        v = m.get("versions") or {}
        s = sess.setdefault(m.get("session"), {
            "model_load_s": m.get("model_load_s"), "warmup_s": m.get("warmup_s"), "repeats": [], "first_repeat": None,
            "gpu": cu.get("gpu") or m.get("device"), "gpu_total_gb": cu.get("total_gb"), "driver": cu.get("driver"),
            "torch_cuda": cu.get("torch_cuda"),
            "versions": {k: v.get(k) for k in ("torch", "sentence_transformers", "transformers", "tokenizers")}})
        s["repeats"].append(r["run"])
        if r.get("first_repeat_in_process"):
            s["first_repeat"] = r["run"]
    J["sessions"] = sess
    m0 = reps[0]["meta"]
    J["gpu"] = (m0.get("cuda") or {}).get("gpu") or m0.get("device")
    J["model"] = m0.get("model")
    # the pass after the repeats, of the same setup, never added to the build totals
    F32 = load_extra(out, FP32_FILE, setup)
    if F32:
        J["fp32_embed"] = {"file": FP32_FILE, "session": F32["meta"].get("session"), "model": F32.get("model"),
                           "model_load_s": F32.get("model_load_s"), "warmup_s": F32.get("warmup_s"),
                           "float32_math": F32.get("float32_math"), "totals": F32.get("totals"),
                           "stores": {c: F32["stores"][c] for c in cells if c in F32["stores"]}}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(J, fh, indent=1)

    a = J["all_stores"]
    L = []
    L.append("# MUSTER build time on BEAM conversation 0 (100K, 1M, 10M), %s" % J["gpu"])
    L.append("")
    if test:
        L.append("TEST ONLY. CPU code-path test with the embedding limited per row. These numbers are not a measurement.")
        L.append("")
    L.append("Repeats complete: %d of %d (%s)." % (len(reps), want, ", ".join(J["files"])))
    if left_out:
        L.append("Left out, saved with another setup (other code, library versions, GPU or stores): %s."
                 % ", ".join(left_out))
    L.append("")
    L.append("Build total = statistics + units + embed + BM25, the four stages of the BEAM harness. The graph stage "
             "(entity index and slot maps) is timed apart and shown as 'with graph'. The paper's build time is the total "
             "with graph.")
    L.append("")
    if r1:
        L.append("Total build time of the %d stores, repeat 1 (the first repeat after the model load and warm-up in its "
                 "process): %.1f s, with graph %.1f s."
                 % (len(cells), a["total_s"]["repeat1"], a["total_with_graph_s"]["repeat1"]))
    else:
        L.append("Repeat 1 is not among the repeats of this setup.")
    tt, tg = a["total_s"]["values"], a["total_with_graph_s"]["values"]
    L.append("Total build time of the %d stores, mean of %d repeats: %.1f s (repeats %s), with graph %.1f s (repeats %s)."
             % (len(cells), len(tt), sum(tt) / len(tt), ", ".join("%.1f s" % x for x in tt), sum(tg) / len(tg),
                ", ".join("%.1f s" % x for x in tg)))
    L.append("First repeat in its process: %s. The other repeats ran warm, after a whole repeat in the same process."
             % (", ".join("repeat %d" % r["run"] for r in reps if r.get("first_repeat_in_process")) or "none"))
    for s, v in sess.items():
        L.append("")
        L.append("Model load: %.2f s from the local files of the pinned revision (the download is not "
                 "included), excluded from every build time (process %s, repeats %s)."
                 % (v["model_load_s"], s, ", ".join(map(str, v["repeats"]))))
        L.append("Warm-up: %.2f s, 64 short texts at batch 32, excluded." % v["warmup_s"])
        vv = v["versions"]
        L.append("GPU and versions of this process: %s, %s GB, driver %s, CUDA %s, torch %s, sentence-transformers %s, "
                 "transformers %s, tokenizers %s." % (v["gpu"], v["gpu_total_gb"], v["driver"], v["torch_cuda"],
                                                      vv.get("torch"), vv.get("sentence_transformers"),
                                                      vv.get("transformers"), vv.get("tokenizers")))
    L.append("")
    L.append("Per store, each time is the mean, then every repeat in brackets (repeat 1 first), in seconds. Units s "
             "includes the two shipped assertions, the re-cut and the non-space check (the table after this one).")
    L.append("")
    def cols(x):                                    # the stages, with repeat 1's build total right after the build total
        out = []
        for k, _ in STAGES:
            out.append(cell(x[k]["values"]))
            if k == "total_s":
                out.append("%.3f" % x[k]["repeat1"] if r1 else "")
        return out

    labs = []
    for k, lab in STAGES:
        labs.append(lab)
        if k == "total_s":
            labs.append("Build total s, repeat 1")
    hdr = ["Store", "Units", "Parents", "Max seq"] + labs
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] + ["---:"] * 3 + ["---:" if lab == "Build total s, repeat 1" else "---" for lab in labs])
             + "|")
    for c in cells:
        x = J["stores"][c]
        L.append("| " + " | ".join([x["name"], "{:,}".format(x["units"]), "{:,}".format(x["parents"]), str(x["emb_maxlen"])]
                                   + cols(x)) + " |")
    L.append("| " + " | ".join(["All %d stores" % len(cells), "{:,}".format(a["units"]), "{:,}".format(a["parents"]), ""]
                               + cols(a)) + " |")
    L.append("")
    L.append("Encoder BAAI/bge-m3 (revision %s), %s, batch 32, normalized."
             % ((J["model"] or {}).get("revision", "")[:7], (J["model"] or {}).get("dtype")))
    L.append("Every store in every repeat matched expected.json (units, parents, the statistics output, the graph "
             "counts, and the sha256 of the unit texts, the embedded texts and the parent ids) before its time was "
             "recorded. expected.json holds the harness's own values (lme_pf29.py with the reported legs' env, on the "
             "CPU).")
    L.append("")
    # the two shipped assertions inside Units s, recorded apart
    L.append("## Inside Units s, the two shipped assertions")
    L.append("")
    L.append("Each time is the mean, then every repeat in brackets, in seconds. The re-cut assertion is lme_pf29.py:2221, "
             "the second cut of the whole text. The non-space check is lme_pf29.py:2227-2230, three re.sub passes (over the "
             "text, over the joined units, and over the text again in the assertion), timed from the start of the first "
             "to the end of the third. Both are inside Units s and the build total above. The last column is Units s "
             "minus the two, repeat by repeat.")
    L.append("")
    L.append("| Store | Units s | Re-cut assertion s | Non-space check s | Units s without the two assertions |")
    L.append("|---|---|---|---|---|")

    def wo(x):
        return [u - rc - ns for u, rc, ns in zip(x["units_s"]["values"], x["units_recut_s"]["values"],
                                                 x["units_nonspace_s"]["values"])]
    for c in cells:
        x = J["stores"][c]
        L.append("| %s | %s | %s | %s | %s |" % (x["name"], cell(x["units_s"]["values"]), cell(x["units_recut_s"]["values"]),
                                                 cell(x["units_nonspace_s"]["values"]), cell(wo(x))))
    L.append("| All %d stores | %s | %s | %s | %s |" % (len(cells), cell(a["units_s"]["values"]),
                                                        cell(a["units_recut_s"]["values"]),
                                                        cell(a["units_nonspace_s"]["values"]), cell(wo(a))))
    L.append("")
    # the fp32 embed pass
    L.append("## fp32 embed pass, the precision of the scored legs (apart)")
    L.append("")
    if F32:
        f = J["fp32_embed"]
        fm = f.get("float32_math") or {}
        L.append("One more embed pass per store after the repeats, with a second model at the same revision and without "
                 ".half(), as lme_pf29.py:2276 loads bge-m3 for the scored BEAM legs, at max_seq 2048 over the same units. "
                 "It is not in any build total above, which all use the fp16 embed.")
        L.append("fp32 model load: %.2f s from the same local files (the download is not included), warm-up %.2f s, both "
                 "excluded. Model dtype %s. float32 matmul precision %s, TF32 for matmul %s and for cuDNN %s, as torch "
                 "starts (the script sets none of them)."
                 % (f["model_load_s"], f["warmup_s"], (f.get("model") or {}).get("dtype"),
                    fm.get("float32_matmul_precision"), "on" if fm.get("cuda_matmul_allow_tf32") else "off",
                    "on" if fm.get("cudnn_allow_tf32") else "off"))
        L.append("")
        L.append("| Store | Texts | Max seq | Embed s, fp16 (mean [repeats]) | Embed s, fp32 (one pass) | fp32 / fp16 mean |")
        L.append("|---|---:|---:|---|---:|---:|")
        e16_all, e32_all = 0.0, 0.0
        for c in cells:
            x, y = J["stores"][c], f["stores"].get(c)
            if not y:
                continue
            e16 = x["embed_s"]["mean"]
            e16_all += e16
            e32_all += y["embed_fp32_s"]
            L.append("| %s | %s | %d | %s | %.3f | %s |" % (x["name"], "{:,}".format(y["texts_embedded_now"]), y["emb_maxlen"],
                                                           cell(x["embed_s"]["values"]), y["embed_fp32_s"],
                                                           ("%.2f" % (y["embed_fp32_s"] / e16)) if e16 > 0 else ""))
        L.append("| All %d stores | | | %.3f | %.3f | %s |" % (len(cells), e16_all, e32_all,
                                                               ("%.2f" % (e32_all / e16_all)) if e16_all > 0 else ""))
    else:
        L.append("Not in this run (no complete %s of this setup)." % FP32_FILE)
    L.append("")
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(L))
    return J


if __name__ == "__main__":
    j = write(sys.argv[1])
    print(open(os.path.join(sys.argv[1], "summary.md")).read() if j else "no complete repeat in %s" % sys.argv[1])
