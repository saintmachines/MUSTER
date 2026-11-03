#!/usr/bin/env python3
"""Write the README's headline blocks and figures from docs/data/headline.json, the one source of their numbers.

Blocks are the text between `<!-- BEGIN headline:NAME -->` and `<!-- END headline:NAME -->` in README.md:
    kpis       the four numbers under the title
    overview   the paper's overview table: six settings, two readers, eight systems
    build      building memory for one 10M-token conversation, as a table
    lead       MemoryAgentBench Overall at the benchmark's reader, as a table
Figures, one file per GitHub theme (the README picks one with <picture>), in docs/assets/readme/:
    score_vs_build_{light,dark}.svg   mean score over the six settings against the time to build a 10M-token memory
    build_cost_{light,dark}.svg       LLM calls, build time and LLM cost of that build
    lead_{light,dark}.svg             where the lead comes from: retrieval and assembly alone, then matched to each store's detected structure

Nothing is typed twice. Before writing, the derived columns of the overview table (per-setting ranks, mean rank, worst rank,
top-3 count, mean score) are recomputed from the scores and must equal the stored ones, and every LLM cost is recomputed
from build tokens and the list prices in the same file and must equal the stored cost.
Run from anywhere:  python tools/make_readme.py            write README.md blocks and the six SVGs
                    python tools/make_readme.py --check    exit 1 if any of them differs from what the data gives
"""
import argparse
import json
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data" / "headline.json"
README = ROOT / "README.md"
ASSETS = ROOT / "docs" / "assets" / "readme"

# GitHub's page colours and neutrals for its two themes; the SVGs carry no background of their own.
THEMES = {
    "light": dict(page="#ffffff", ink="#1f2328", ink2="#424a53", muted="#59636e", grid="#e6e9ed", axis="#c8d1da",
                  ctx="#8c959f", accent="#0969da", tint="#ddf4ff"),
    "dark": dict(page="#0d1117", ink="#f0f6fc", ink2="#d1d7e0", muted="#9198a1", grid="#21262d", axis="#3d444d",
                 ctx="#6e7681", accent="#4493f8", tint="#0f2238"),
}
FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif'

# Helvetica/Arial advance widths in 1/1000 em (others count as 600); used to keep
# every label inside the canvas and clear of the others.
_REG = dict(zip(" .,-–·:+()×≥$/%'’0123456789", [278, 278, 278, 333, 556, 278, 278, 584, 333, 333, 584, 549, 556, 278, 889,
                                                191, 222] + [556] * 10))
_REG.update(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
                                               667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611]))
_REG.update(zip("abcdefghijklmnopqrstuvwxyz", [556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
                                               556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500]))
_BOLD = dict(_REG)
_BOLD.update(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
                                                667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611]))
_BOLD.update(zip("abcdefghijklmnopqrstuvwxyz", [556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
                                                611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500]))


def text_width(s, size, bold=False):
    table = _BOLD if bold else _REG
    return sum(table.get(ch, 600) for ch in s) * size / 1000 * 1.04


def dec(x, places):
    """Round half up, as the paper's tables do (float formatting would round 1.965 down)."""
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def calls_label(c, long=False):
    if isinstance(c, list):
        return f"{c[0]:,}–{c[1]:,}" if long else f"{c[0] / 1000:.1f}–{c[1] / 1000:.1f}K"
    return f"{c:,}"


def millions(tokens):
    """Build tokens as the table prints them: the stored precision, three decimals below one million."""
    t = Decimal(tokens)
    return f"{t:.3f}M" if t < 1 else f"{tokens}M"


def seconds(label):
    v, unit = re.match(r"≈?([\d.,]+)\s*(s|min|h)$", label).groups()
    return float(v.replace(",", "")) * {"s": 1, "min": 60, "h": 3600}[unit]


# ---------------------------------------------------------------------------------------------------- derived values

def derive(d):
    """Recompute what the paper derives and fail on any difference; return the values the outputs need."""
    rows = d["overview"]["rows"]
    n = len(d["settings"])
    ranks = {r["system"]: [] for r in rows}
    for j in range(n):
        order = sorted(rows, key=lambda r: -r["scores"][j])
        for i, r in enumerate(order):
            ranks[r["system"]].append(i + 1)
    for r in rows:
        k = ranks[r["system"]]
        want = {"mean_rank": dec(sum(k) / n, 2), "worst_rank": max(k), "top3": sum(1 for x in k if x <= 3)}
        got = {"mean_rank": dec(r["mean_rank"], 2), "worst_rank": r["worst_rank"], "top3": r["top3"]}
        if want != got:
            sys.exit(f"{r['system']}: stored {got} but the scores give {want}")
        mean = sum(Decimal(str(s)) for s in r["scores"]) / n
        if abs(mean - Decimal(str(r["mean_score"]))) > Decimal("0.0051"):   # the paper averages unrounded scores
            sys.exit(f"{r['system']}: mean score {r['mean_score']} but the scores average {mean}")
    us = next(r for r in rows if r.get("us"))
    full = next(r for r in rows if r.get("memory_system") is False)
    others = [r for r in rows if not r.get("us")]

    prices = d["build_10m"]["prices_usd_per_1m"]
    cost = {}
    for b in d["build_10m"]["rows"]:
        if abs(seconds(b["build"]) - b["build_s"]) > 0.02 * b["build_s"]:
            sys.exit(f"{b['system']}: build label {b['build']} disagrees with {b['build_s']} s")
        if b["model"] is None:
            cost[b["system"]] = "$0"
            continue
        p = prices[b["model"]]
        usd = Decimal(b["tokens_in"]) * Decimal(str(p["in"]))
        lower_bound = "out" in p and b.get("tokens_out") is None
        if b.get("tokens_out") is not None:
            usd += Decimal(b["tokens_out"]) * Decimal(str(p["out"]))
        cost[b["system"]] = ("≥ " if lower_bound else "") + f"${dec(usd, 2)}"
    for b in d["build_10m"]["rows"]:
        if cost[b["system"]] != b["llm_cost"]:
            sys.exit(f"{b['system']}: stored LLM cost {b['llm_cost']} but tokens x prices give {cost[b['system']]}")
    build = {b["system"]: b for b in d["build_10m"]["rows"]}
    for r in rows:   # the two tables give the same build calls
        if r["system"] in build and r["build_calls"] != build[r["system"]]["calls"]:
            sys.exit(f"{r['system']}: build calls differ between the overview and the build table")
    nxt = min((b for b in d["build_10m"]["rows"] if b["system"] != us["system"]), key=lambda b: b["build_s"])
    return dict(rows=rows, ranks=ranks, us=us, full=full, others=others, cost=cost, build=build,
                speedup=round(nxt["build_s"] / build[us["system"]]["build_s"]), next_fastest=nxt["system"],
                read_ratio=round(full["read_usd_per_1k"] / us["read_usd_per_1k"]),
                next_score=max(r["mean_score"] for r in others))


# ---------------------------------------------------------------------------------------------------- README blocks

def block_kpis(d, v):
    us = v["us"]
    cells = [
        (f"{us['mean_rank']:.2f}", f"mean rank of {len(v['rows'])} over six settings,<br>two frontier readers"),
        (f"{dec(us['mean_score'], 1)}", f"mean score over the six;<br>next best {dec(v['next_score'], 1)}"),
        (f"1/{v['read_ratio']}", "the full context's read cost<br>per 1,000 questions"),
        (f"{us['build_calls']} · {v['build'][us['system']]['build']}", "LLM calls and time to build<br>a 10M-token memory"),
    ]
    out = ['<table align="center">', "<tr>"]
    out += [f'<td align="center"><b>{a}</b><br><sub>{b}</sub></td>' for a, b in cells]
    return "\n".join(out + ["</tr>", "</table>"])


def block_overview(d, v):
    rows, ranks = v["rows"], v["ranks"]
    head = ["System"] + [f"{s['bench'].replace('LoCoMo-Refined', 'LoCoMo-Ref.')}<br>{s['reader']}" for s in d["settings"]]
    head += ["Mean<br>rank", "Mean<br>score", "Worst<br>rank", "Top 3<br>(of 6)", "LLM calls<br>10M build"]
    out = ["| " + " | ".join(head) + " |", "|:--" + "|--:" * (len(head) - 1) + "|"]
    for r in rows:
        name = r["system"].replace(" ", "&nbsp;").replace("-", "\u2011")   # keep names on one line (no-break space, no-break hyphen)
        name += "\\*" if r.get("memory_system") is False else ""
        cells = [name] + [f"{s:.2f}" for s in r["scores"]]
        cells += [f"{r['mean_rank']:.2f}", f"{r['mean_score']:.2f}", str(r["worst_rank"]), str(r["top3"]),
                  calls_label(r["build_calls"])]
        if r.get("us"):   # bold: MUSTER's whole row, as the paper's overview table marks it
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def block_build(d, v):
    out = ["| System | LLM calls | Build time | LLM cost | Notes |", "|:--|--:|--:|--:|:--|"]
    prices = d["build_10m"]["prices_usd_per_1m"]
    for b in build_order(d):
        if b["model"] is None:
            how = "no generative call; built " + b["build_note"]
        else:
            p = prices[b["model"]]
            parts = [f"{millions(b['tokens_in'])} input tokens at ${p['in']:.2f}/M"]
            if b.get("tokens_out") is not None:
                parts.append(f"{millions(b['tokens_out'])} output tokens at ${p['out']:.2f}/M")
            elif "out" in p:
                parts.append("output not recorded")
            how = f"cost: {b['model']} ({p['where']}), " + ", ".join(parts)
            if b.get("tokens_note"):
                how += "; " + b["tokens_note"]
            if b["build_note"]:
                how = f"build time: {b['build_note']}; " + how
        name = f"**{b['system']}**" if b["system"] == v["us"]["system"] else b["system"]
        out.append(f"| {name} | {calls_label(b['calls'], long=True)} | {b['build']} | {v['cost'][b['system']]} | {how} |")
    full = v["full"]
    out.append(f"| {full['system']} | 0 | nothing to build | $0 | reads everything for every question, "
               f"{full['context'].split('/')[-1]} tokens per BEAM-10M question |")
    return "\n".join(out)


def block_lead(d, v):
    out = ["| | MemoryAgentBench Overall | What it is |", "|:--|--:|:--|"]
    for r in d["lead"]["rows"]:
        name = f"**{r['label']}**" if r["kind"] == "us" else r["label"]
        out.append(f"| {name} | {r['overall']:.1f} | {r['note']} |")
    return "\n".join(out)


BLOCKS = {"kpis": block_kpis, "overview": block_overview, "build": block_build, "lead": block_lead}


def fill_readme(text, d, v):
    for name, fn in BLOCKS.items():
        pat = re.compile(r"(<!-- BEGIN headline:%s -->\n)(?:.*?\n)?(<!-- END headline:%s -->)" % (name, name), re.S)
        if len(pat.findall(text)) != 1:
            sys.exit(f"README.md needs exactly one headline:{name} block")
        text = pat.sub(lambda m: m.group(1) + fn(d, v) + "\n" + m.group(2), text)
    return text


# ---------------------------------------------------------------------------------------------------- SVG helpers

class Canvas:
    def __init__(self, w, h, title, desc, c):
        self.w, self.h, self.c, self.boxes = w, h, c, []
        self.out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                    f'role="img" aria-labelledby="t d">',
                    f'<title id="t">{esc(title)}</title>', f'<desc id="d">{esc(desc)}</desc>',
                    f"<style>text{{font-family:{FONT}}}</style>"]

    def add(self, s):
        self.out.append(s)

    def text(self, x, y, s, size, fill, anchor="start", bold=False, track=False, rotate=None):
        w = text_width(s, size, bold)
        left = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
        if rotate is None:
            if left < 0 or left + w > self.w + 0.5:
                sys.exit(f"label outside the canvas: {s!r}")
            self.boxes.append((left, y - size * 0.8, left + w, y + size * 0.25, s))
        attrs = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"'
        if anchor != "start":
            attrs += f' text-anchor="{anchor}"'
        if bold:
            attrs += ' font-weight="700"'
        if track:
            attrs += ' letter-spacing=".02em"'
        if rotate is not None:
            attrs += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.add(f"<text {attrs}>{esc(s)}</text>")

    def done(self):
        b = self.boxes
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                if b[i][0] < b[j][2] and b[j][0] < b[i][2] and b[i][1] < b[j][3] and b[j][1] < b[i][3]:
                    sys.exit(f"labels overlap: {b[i][4]!r} and {b[j][4]!r}")
        return "\n".join(self.out + ["</svg>"]) + "\n"


def bar_path(x0, x1, y, h, r=4):
    """A bar anchored at x0 with a rounded data end at x1."""
    return (f"M{x0:.1f} {y:.1f} H{x1 - r:.1f} a{r} {r} 0 0 1 {r} {r} V{y + h - r:.1f} "
            f"a{r} {r} 0 0 1 -{r} {r} H{x0:.1f} Z")


# ---------------------------------------------------------------------------------------------------- figures

def fig_score_vs_build(d, v, c):
    W, H, ml, mr, mt, mb = 720, 380, 48, 24, 18, 46
    lo, hi, y0, y1 = 10, 2e5, 45, 75

    def X(t):
        return ml + (math.log10(t) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - ml - mr)

    def Y(s):
        return H - mb - (s - y0) / (y1 - y0) * (H - mt - mb)

    pts = [(r, v["build"][r["system"]]) for r in v["rows"] if r["system"] in v["build"]]
    desc = "; ".join(f"{r['system']} {r['mean_score']:.2f} in {b['build']}" + (f" ({b['build_note']})" if b.get("build_mark") else "")
                     for r, b in pts)
    desc += f"; {v['full']['system']} {v['full']['mean_score']:.2f} with nothing to build"
    k = Canvas(W, H, "Mean score over six settings against the time to build a 10M-token memory, log time axis",
               desc + ".", c)
    k.add(f'<g stroke="{c["grid"]}" stroke-width="1">')
    for s in range(50, 80, 5):
        k.add(f'<line x1="{ml}" x2="{W - mr}" y1="{Y(s):.1f}" y2="{Y(s):.1f}"/>')
    k.add("</g>")
    k.add(f'<line x1="{ml}" x2="{W - mr}" y1="{H - mb}" y2="{H - mb}" stroke="{c["axis"]}" stroke-width="1"/>')
    for s in range(50, 80, 5):
        k.text(ml - 8, Y(s) + 4, str(s), 11, c["muted"], "end")
    for t, lab in [(10, "10 s"), (60, "1 min"), (600, "10 min"), (3600, "1 h"), (36000, "10 h")]:
        k.text(X(t) - text_width(lab.split()[0], 11) / 2, H - mb + 18, lab, 11, c["muted"])  # numeral centred on its time
    k.text((ml + W - mr) / 2, H - 8, "time to build one 10M-token conversation, as run (log scale)", 11, c["muted"], "middle")
    k.text(14, (mt + H - mb) / 2, "mean score, six settings", 11, c["muted"], "middle", rotate=-90)
    full = v["full"]
    yf = Y(full["mean_score"])
    k.add(f'<line x1="{ml}" x2="{W - mr}" y1="{yf:.1f}" y2="{yf:.1f}" stroke="{c["ctx"]}" stroke-width="1.5" '
          f'stroke-dasharray="5 4"/>')
    k.text(ml + 8, yf + 15, f"{full['system']}: reads everything, nothing to build", 11, c["muted"])
    off = {"MUSTER": (12, -4, "start"), "True Memory": (-12, 4, "end"), "HippoRAG-v2": (12, 4, "start"),
           "EverMemOS": (8, 21, "end"), "Mem0": (-12, 4, "end"), "Hindsight": (-12, 4, "end"), "Honcho": (12, 4, "start")}
    for r, b in sorted(pts, key=lambda p: bool(p[0].get("us"))):
        cx, cy, us = X(b["build_s"]), Y(r["mean_score"]), bool(r.get("us"))
        k.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{8 if us else 6}" fill="{c["accent"] if us else c["ctx"]}" '
              f'stroke="{c["page"]}" stroke-width="2"/>')
        dx, dy, anchor = off[r["system"]]
        label = f"{r['system']} · {b['build']}" if us else r["system"] + (f" · {b['build_mark']}" if b.get("build_mark") else "")
        k.text(cx + dx, cy + dy, label, 13 if us else 12, c["ink"] if us else c["ink2"], anchor, bold=us)
        if us:
            k.text(cx + dx + text_width(label, 13, True) + 10, cy + dy, "no per-store tuning", 11, c["muted"])
            k.text(cx + 12, cy + 14, f"{v['speedup']}× faster than the next fastest", 11, c["muted"], bold=True)
    return k.done()


def build_order(d):
    """Systems that build with an LLM first, most calls (then longest build) first; those that store text as is last."""
    def hi(b):
        return b["calls"][1] if isinstance(b["calls"], list) else b["calls"]
    rows = sorted(d["build_10m"]["rows"], key=lambda b: (-hi(b), -b["build_s"]))
    return [b for b in rows if hi(b) > 0] + [b for b in rows if hi(b) == 0]


def fig_build_cost(d, v, c):
    W, bh, gap, gh, ml, mr, mt, mb = 720, 22, 12, 22, 130, 16, 32, 56
    rows = build_order(d)
    groups = [("Extract with an LLM", [b for b in rows if b["calls"] != 0]), ("Store as is", [b for b in rows if b["calls"] == 0])]
    H = mt + mb + len(groups) * gh + len(rows) * (bh + gap) - gap
    bx1, ct, cu, top = 430, 548, W - mr, 30000

    def X(n):
        return ml + n / top * (bx1 - ml)

    desc = "; ".join(f"{b['system']} {calls_label(b['calls'], long=True)} LLM calls, {b['build']}"
                     f"{' (' + b['build_note'] + ')' if b.get('build_mark') else ''}, LLM cost "
                     f"{v['cost'][b['system']]}" for b in rows)
    k = Canvas(W, H, "Building memory for one 10M-token conversation: LLM calls, build time and LLM cost", desc + ".", c)
    k.add(f'<g stroke="{c["grid"]}" stroke-width="1">')
    for n in range(0, top + 1, 10000):
        k.add(f'<line x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{mt - 6}" y2="{H - mb}"/>')
    k.add("</g>")
    for x, lab, anchor in [(ml, "LLM calls", "start"), (ct, "build time", "end"), (cu, "LLM cost", "end")]:
        k.text(x, mt - 14, lab, 11, c["muted"], anchor)
    for n in range(0, top + 1, 10000):
        k.text(X(n), H - mb + 16, "0" if n == 0 else f"{n // 1000}K", 11, c["muted"], "middle")
    y = mt
    for title, members in groups:
        k.text(10, y + 14, title, 11, c["muted"], bold=True, track=True)
        y += gh
        for b in members:
            us = b["system"] == v["us"]["system"]
            cy = y + bh / 2 + 4
            k.text(ml - 10, cy, b["system"], 13 if us else 12, c["ink"] if us else c["ink2"], "end", bold=us)
            if isinstance(b["calls"], list):
                a, z = X(b["calls"][0]), X(b["calls"][1])
                k.add(f'<rect x="{a:.1f}" y="{y + 4}" width="{z - a:.1f}" height="{bh - 8}" rx="4" fill="{c["ctx"]}" '
                      f'opacity=".55"/>')
                k.text(z + 8, cy, calls_label(b["calls"]), 12, c["ink2"])
            elif b["calls"] == 0:
                k.add(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{y + 3}" y2="{y + bh - 3}" '
                      f'stroke="{c["accent"] if us else c["ctx"]}" stroke-width="3" stroke-linecap="round"/>')
                k.text(X(0) + 10, cy, "0", 13 if us else 12, c["ink"] if us else c["ink2"], bold=us)
            else:
                k.add(f'<path d="{bar_path(X(0), X(b["calls"]), y + 4, bh - 8)}" fill="{c["ctx"]}"/>')
                k.text(X(b["calls"]) + 8, cy, calls_label(b["calls"]), 12, c["ink2"])
            k.text(ct, cy, b["build"], 13 if us else 12, c["ink"] if us else c["ink2"], "end", bold=us)
            if b.get("build_mark"):   # a build time that is not one conversation's says so under it
                k.text(ct, cy + 14, b["build_mark"], 11, c["muted"], "end")
            k.text(cu, cy, v["cost"][b["system"]], 13 if us else 12, c["ink"] if us else c["ink2"], "end", bold=us)
            y += bh + gap
    full, us = v["full"], v["us"]
    k.text(10, H - 12, f"{full['system']} reads everything: nothing to build, up to "
                       f"{full['context'].split('/')[-1]} tokens per question (${full['read_usd_per_1k']:.2f} against "
                       f"${us['read_usd_per_1k']:.2f} per 1,000 questions).", 11, c["muted"])
    return k.done()


def fig_lead(d, v, c):
    rows = d["lead"]["rows"]
    W, bh, gap, ml, mr, mt, mb = 720, 24, 14, 250, 182, 12, 44
    H = mt + mb + len(rows) * (bh + gap) - gap

    def X(s):
        return ml + s / 60 * (W - ml - mr)

    desc = "; ".join(f"{r['label']} {r['overall']:.1f}" for r in rows)
    k = Canvas(W, H, "MemoryAgentBench Overall at the benchmark's reader, GPT-4o-mini: where MUSTER's lead comes from",
               desc + ".", c)
    k.add(f'<g stroke="{c["grid"]}" stroke-width="1">')
    for s in range(0, 60, 10):
        k.add(f'<line x1="{X(s):.1f}" x2="{X(s):.1f}" y1="{mt - 4}" y2="{H - mb}"/>')
    k.add("</g>")
    for s in range(0, 60, 10):
        k.text(X(s), H - mb + 16, str(s), 11, c["muted"], "middle")
    k.text((ml + X(50)) / 2, H - 6, "MemoryAgentBench Overall, GPT-4o-mini reader", 11, c["muted"], "middle")
    best_other = max(r["overall"] for r in rows if r["kind"] == "other")
    prev = None
    for i, r in enumerate(rows):
        y = mt + i * (bh + gap)
        cy, x1, us = y + bh / 2 + 4, X(r["overall"]), r["kind"] == "us"
        k.text(ml - 10, cy, r["label"], 13 if us else 12, c["ink"] if us else c["ink2"], "end", bold=us)
        path = bar_path(X(0), x1, y, bh)
        if r["kind"] == "other":
            k.add(f'<path d="{path}" fill="{c["ctx"]}"/>')
        elif r["kind"] == "step":
            k.add(f'<path d="{path}" fill="{c["tint"]}" stroke="{c["accent"]}" stroke-width="1.5"/>')
        else:
            k.add(f'<path d="{path}" fill="{c["accent"]}"/>')
        value = f"{r['overall']:.1f}"
        k.text(x1 + 8, cy, value, 13 if r["kind"] != "other" else 12, c["ink"] if r["kind"] != "other" else c["ink2"],
               bold=r["kind"] != "other")
        if r["kind"] != "other":
            delta = (f"+{dec(r['overall'] - best_other, 1)} over the best other" if prev is None
                     else f"+{dec(r['overall'] - prev, 1)}")
            k.text(x1 + 8 + text_width(value, 13, True) + 10, cy, delta, 11, c["muted"], bold=True)
            prev = r["overall"]
    return k.done()


FIGURES = {"score_vs_build": fig_score_vs_build, "build_cost": fig_build_cost, "lead": fig_lead}


# ---------------------------------------------------------------------------------------------------- main

def outputs():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    v = derive(d)
    files = {README: fill_readme(README.read_text(encoding="utf-8"), d, v)}
    for name, fn in FIGURES.items():
        for theme, c in THEMES.items():
            files[ASSETS / f"{name}_{theme}.svg"] = fn(d, v, c)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing; exit 1 on any difference")
    args = ap.parse_args()
    stale = []
    for path, text in outputs().items():
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old == text:
            continue
        stale.append(path.relative_to(ROOT).as_posix())
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if args.check:
        print("stale:", stale or "none")
        sys.exit(1 if stale else 0)
    print("written:", stale or "nothing changed")


if __name__ == "__main__":
    main()
