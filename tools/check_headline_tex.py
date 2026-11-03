#!/usr/bin/env python3
"""Compare docs/data/headline.json with the paper's tex source, cell by cell.

Usage: python tools/check_headline_tex.py PAPER.tex
Checks the overview table (tab:overview: every score and summary cell of the eight rows), the build table of one 10M-token
conversation (tab:beamcost: calls, build time, LLM input tokens at the precision the paper prints), the bars of the
"How it works" figure (tab:board and tab:gate) and the mechanism block (every tab:gate cell, the floors of Algorithm 1,
the FC-SH order rows of tab:slotsrc). Prints one line per compared cell group and exits 1 on any difference.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs" / "data" / "headline.json").read_text(encoding="utf-8"))


def unwrap(s, cmd):
    """Replace every \\cmd{...} by its content, braces matched."""
    key = "\\" + cmd + "{"
    while key in s:
        i = s.index(key)
        j, depth = i + len(key), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            j += 1
        s = s[:i] + s[i + len(key):j - 1] + s[j:]
    return s


def clean(cell):
    cell = re.sub(r"\\cite\{[^}]*\}|\\smash|\\phantom\{0\}|\\ours", "", cell)
    cell = unwrap(cell, "textbf").replace("\\sys{}", "MUSTER")
    cell = cell.replace("{,}", ",").replace("$", "").replace("\\,", "").replace("--", "–").replace("^\\ast", "")
    return cell.replace("\\\\", "").replace("{", "").replace("}", "").strip()


def table(tex, label):
    i = tex.index("\\label{%s}" % label)
    return tex[i:tex.index("\\end{tabular}", i)]


def rows(block):
    return [[clean(c) for c in line.split("&")] for line in block.splitlines() if line.count("&") >= 2]


def main():
    tex = Path(sys.argv[1]).read_text(encoding="utf-8")
    bad, n = [], 0

    def same(what, got, want):
        nonlocal n
        n += 1
        if got != want:
            bad.append(f"{what}: headline {got!r}, paper {want!r}")

    # overview table: tex names to headline names
    names = {"MUSTER": "MUSTER", "Full store": "Full context", "True Memory": "True Memory", "Mem0": "Mem0",
             "EverMemOS": "EverMemOS", "HippoRAG-v2": "HippoRAG-v2", "Hindsight": "Hindsight", "Honcho": "Honcho"}
    tex_rows = {names[r[0]]: r for r in rows(table(tex, "tab:overview")) if r[0] in names}
    sentence_layout = False
    for r in DATA["overview"]["rows"]:
        t = tex_rows[r["system"]]
        calls = r["build_calls"]
        calls = f"{calls[0] / 1000:.1f}–{calls[1] / 1000:.1f}K" if isinstance(calls, list) else f"{calls:,}"
        mine = [f"{s:.2f}" for s in r["scores"]] + [f"{r['mean_rank']:.2f}", f"{r['mean_score']:.2f}", str(r["worst_rank"])]
        if len(t) == len(mine) + 3:   # v209 on: Top-3 as k/6, no context or read-cost column
            sentence_layout = True
            mine += [f"{r['top3']}/{len(r['scores'])}", calls]
        else:
            mine += [str(r["top3"]), r["context"], f"{r['read_usd_per_1k']:.2f}", calls]
        same(f"overview {r['system']}", mine, t[1:])
    printed = re.sub(r"(?<!\\)%.*", "", tex)   # a LaTeX comment runs from an unescaped % to the end of the line
    reads = re.findall(r"\\\$([\d.]+) for \\sys\{\} and \\\$([\d.]+) for the full", printed)
    if sentence_layout and len(reads) != 1:
        bad.append(f"overview: the read-cost columns are gone and the printed text holds {len(reads)} read-cost sentences, not one")
    if len(reads) == 1:   # v209 on, the read costs are a sentence
        cost = {r["system"]: f"{r['read_usd_per_1k']:.2f}" for r in DATA["overview"]["rows"]}
        same("read cost per 1,000 questions", [cost["MUSTER"], cost["Full context"]], list(reads[0]))
    print(f"overview: {len(DATA['overview']['rows'])} rows of {len(tex_rows)} compared")

    # build table, 10M column
    bt = table(tex, "tab:beamcost")
    blocks = {}
    current = None
    for line in bt.splitlines():
        if "&" not in line:
            continue
        cells = [clean(c) for c in line.split("&")]
        if cells[0] and not cells[0].startswith("\\quad"):  # an indented line names the writer, not a system
            current = cells[0].split("^")[0].strip()
        blocks.setdefault(current, {})[cells[1]] = cells[-1]
    LM_CALLS, LM_INPUT = ("language-model calls", "LM calls"), ("model input ( times source)", "LM input ( times source)")
    spec = {  # headline system: (tex block, calls row, build-time row, input row); a tuple names a row across versions
        "MUSTER": ("MUSTER", "generative calls / input tokens", ("build time, bge-m3 on one GPU", "build time, one GPU^dagger"), None),
        "True Memory": ("True Memory", "generative calls / input tokens", "build time, one GPU", None),
        "Mem0": ("Mem0 2.1.0", LM_CALLS, "wall-clock", LM_INPUT),
        "Honcho": ("Honcho", "model calls (estimated^S)", "wall-clock (with queue drain)", None),
        "Hindsight": ("Hindsight", LM_CALLS, "per conversation (mean)", LM_INPUT),
        "EverMemOS": ("EverMemOS", LM_CALLS, ("per conversation (mean), all at once", "wall-clock, whole scale"), LM_INPUT),
        "HippoRAG-v2": ("HippoRAG-v2", LM_CALLS, "per conversation (mean)", LM_INPUT),
    }

    def find(block, row):
        b = blocks[block]
        for name in row if isinstance(row, tuple) else (row,):
            key = name.replace("^dagger", "").replace("^S", "").replace(" times", "")
            for k, val in b.items():
                if k.replace("^\\dagger", "").replace("^\\S", "").replace("^\\|", "").replace("\\times", "").replace("  ", " ").strip() == key.strip():
                    return val
        raise KeyError(f"{block}: {row}")

    def secs(t):
        v, u = re.match(r"([\d.,]+)\s*(s|min|h)", t).groups()
        return float(v.replace(",", "")) * {"s": 1, "min": 60, "h": 3600}[u]

    for b in DATA["build_10m"]["rows"]:
        block, crow, trow, irow = spec[b["system"]]
        c = find(block, crow).split("/")[0].strip()
        calls = f"{b['calls'][0]:,}–{b['calls'][1]:,}" if isinstance(b["calls"], list) else f"{b['calls']:,}"
        same(f"build calls {b['system']}", calls, c)
        t = secs(find(block, trow))
        same(f"build time {b['system']}", round(b["build_s"], 1), round(t, 1))
        if irow:
            cell = find(block, irow)
            num = re.search(r"([\d.]+)M", cell).group(1)
            places = len(num.split(".")[1]) if "." in num else 0
            same(f"build input {b['system']}", f"{float(b['tokens_in']):.{places}f}", num)
    honcho = next(b for b in DATA["build_10m"]["rows"] if b["system"] == "Honcho")
    sent = re.search(r"([\d.]+)M sent", honcho["tokens_note"]).group(1)
    same("Honcho tokens sent per conversation", f"{float(sent) / 10:.1f}M", find("Honcho", "tokens sent"))
    print(f"build: {len(DATA['build_10m']['rows'])} systems compared")

    # bars of the "How it works" figure
    board = {r[0]: r[-1] for r in rows(table(tex, "tab:board"))}
    gate = {r[0]: r[-1] for r in rows(table(tex, "tab:gate"))}
    want = {"BM25": board["BM25"], "Full context (128K window)": board["GPT-4o-mini 128K"],
            "True Memory": board["True Memory"],
            "Retrieval + assembly, no detection": gate["Hybrid retrieval and assembly, benchmark chunks"],
            "+ detection: units, parents, order": gate["+ detection, \\balanced everywhere"],
            "+ retrieval mix matched (MUSTER)": gate["+ detection with its gate (MUSTER)"]}
    for r in DATA["lead"]["rows"]:
        same(f"lead {r['label']}", f"{r['overall']:.1f}", want[r["label"]])
    print(f"lead: {len(DATA['lead']['rows'])} bars compared")

    # mechanism: every cell of the gate table, the two floor triples of Algorithm 1, the order rows of the slot-source table
    mech = DATA["mechanism"]
    gate_rows = {"none": "Hybrid retrieval and assembly, benchmark chunks", "balanced": "+ detection, \\balanced everywhere",
                 "parent_priority": "+ detection, \\parprio everywhere", "muster": "+ detection with its gate (MUSTER)"}
    full = {r[0]: r[1:] for r in rows(table(tex, "tab:gate"))}
    for g in mech["gate"]:
        same(f"gate {g['key']}", [f"{x:.1f}" for x in g["values"]], full[gate_rows[g["key"]]])
    alg = re.search(r"parent-priority \$\(([\d,]+)\)\$ if .*?else balanced \$\(([\d,]+)\)\$", tex)
    same("floors", [mech["floors"]["parent_priority"], mech["floors"]["balanced"]],
         [[int(x) for x in alg.group(1).split(",")], [int(x) for x in alg.group(2).split(",")]])
    slot = {r[0]: r[-1] for r in rows(table(tex, "tab:slotsrc"))}
    want_order = {"recorded": "Read from the bytes (MUSTER)", "none": "No order: all records, no serials, random order",
                  "shuffled": "Recorded serials shuffled among the records"}
    for o in mech["order_fc_sh"]:
        same(f"order {o['order']}", str(o["fc_sh"]), slot[want_order[o["order"]]])
    words = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                         "fourteen fifteen sixteen".split())}
    run = re.search(r"over (\w+) stores they run\s+in (\w+) seconds", tex)
    same("detection run", [mech["detect"]["stores"], mech["detect"]["seconds"]], [words[run.group(1)], words[run.group(2)]])
    print(f"mechanism: {len(mech['gate'])} gate rows, 2 floor triples, {len(mech['order_fc_sh'])} order rows, the detection run compared")

    # detection statistics: each store against the gate report, the thresholds against the rule, the gaps against Figure 1
    ds = DATA["detect_stats"]
    gate_md = (ROOT / "gate" / "stage0_derive.md").read_text(encoding="utf-8")
    printed = dict((k, (float(a), float(b))) for k, a, b in
                   re.findall(r"(\w+) (\d\.\d{4}) / (\d\.\d{4})", gate_md.split("Statistics printed by the rule")[1].split("The paper's")[0]))
    same("detection stores", sorted(s["key"] for s in ds["stores"]), sorted(printed))
    for s in ds["stores"]:
        same(f"detection {s['key']}", (s["recshare"], s["serial"]), printed[s["key"]])
    rule = re.search(r"If \$\\serial \\ge ([\d.]+)\$.*?when \$\\cov \\ge ([\d.]+)\$", tex, re.S)
    same("detection thresholds", [f"{ds['tau']['serial']:.2f}", f"{ds['tau']['recshare']:.2f}"], [rule.group(1), rule.group(2)])
    cap = re.search(r"\$([\d.]+)\$ wide for \\serial\{\} and \$([\d.]+)\$ for \\cov\{\}", tex)
    same("detection gaps", [f"{ds['gap']['serial']:.3f}", f"{ds['gap']['recshare']:.3f}"], [cap.group(1), cap.group(2)])
    for stat in ("serial", "recshare"):
        v = [s[stat] for s in ds["stores"]]
        hi = min(x for x in v if x >= ds["tau"][stat])
        lo = max(x for x in v if x < ds["tau"][stat])
        same(f"detection gap from the stores, {stat}", f"{hi - lo:.3f}", f"{ds['gap'][stat]:.3f}")
    print(f"detection: {len(ds['stores'])} stores, 2 thresholds, 2 gaps compared")

    # MemoryAgentBench competency means (tab:board, last five columns)
    names = {"MUSTER": ("MUSTER (ours)", "MUSTER, GPT-4o-mini"), "True Memory": "True Memory",
             "GPT-4o-mini 128K": "GPT-4o-mini 128K", "HippoRAG-v2": "HippoRAG-v2", "BM25": "BM25", "Honcho": "Honcho",
             "Hindsight": "Hindsight 0.10.3", "Mem0 2.1.0": "Mem0 2.1.0, full-ver.\\", "EverMemOS": "EverMemOS^\\dagger"}
    board_rows = {re.sub(r"\s+,", ",", r[0]): r[-5:] for r in rows(table(tex, "tab:board"))}
    for r in DATA["mab_board"]["rows"]:
        key = names[r["system"]]
        if isinstance(key, tuple):   # the paper renamed MUSTER's GPT-4o-mini row when Tables 1 and 2 merged
            key = next((k for k in key if k in board_rows), key[0])
        if key not in board_rows:
            sys.exit(f"tab:board has no row {key!r}; its rows: {sorted(board_rows)}")
        same(f"board {r['system']}", [f"{x:.1f}" for x in r["values"]], board_rows[key])
    print(f"board: {len(DATA['mab_board']['rows'])} rows compared")

    # LoCoMo and LoCoMo-Refined (tab:locomo)
    names = {"MUSTER": "MUSTER, U=10", "EverMemOS": "EverMemOS", "True Memory": "True Memory, HyDE", "Mem0": "Mem0 2.1.0",
             "HippoRAG-v2": "HippoRAG-v2", "Hindsight": "Hindsight 0.10.3", "Honcho": "Honcho"}
    loc_rows = {re.sub(r"\s+,", ",", r[0]): r for r in rows(table(tex, "tab:locomo"))}   # a dropped \cite leaves " ,"
    for r in DATA["locomo"]["rows"]:
        key = names[r["system"]]
        if key not in loc_rows:
            sys.exit(f"tab:locomo has no row {key!r}; its rows: {sorted(loc_rows)}")
        t = [re.sub(r"\s*\[\d+pt\]$", "", c.replace("^\\dagger", "").replace("^\\S", "")) for c in loc_rows[key]]
        if len(t) >= 10:   # LoCoMo and Refined each with their own context and calls columns
            scores, calls, mine = t[1:3] + t[5:8], [t[4], t[9]], r["calls"]
        else:              # one shared context column and one "LoCoMo/Refined" calls cell
            scores, calls, mine = t[1:6], [t[-1]], ["/".join(dict.fromkeys(r["calls"]))]
        same(f"locomo {r['system']}", [f"{x:.2f}" for x in r["locomo"] + r["refined"]] + mine, scores + calls)
    q = re.findall(r"\\textbf\{LoCoMo(?:-Refined)?\}, ([\d,]+) questions", table(tex, "tab:locomo"))
    same("locomo questions", [f"{DATA['locomo']['questions'][k]:,}" for k in ("LoCoMo", "LoCoMo-Refined")], q)
    print(f"locomo: {len(DATA['locomo']['rows'])} rows compared")

    # Mem0's memories read by MUSTER (tab:extend, BEAM mean)
    ext = {(r[0], r[1]): r[6] for r in rows(table(tex, "tab:extend")) if len(r) >= 8}
    e = DATA["extend"]
    same("extend", [f"{e['mem0_search']:.1f}", f"{e['muster_over_mem0']:.1f}", f"{e['both']:.1f}"],
         [ext.get(("–", "Mem0's search")), ext.get(("–", "MUSTER")), ext.get(("MUSTER, U=30", "MUSTER"))])
    print("extend: 3 means compared")

    print(f"cell groups compared: {n}; differences: {len(bad)}")
    for line in bad:
        print("  DIFF", line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
