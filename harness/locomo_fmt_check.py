"""derive_config.stats() on two other serializations of LoCoMo: (N) LoCoMo's own baseline format (task_eval: per session
'DATE: ...\nCONVERSATION:\n' + 'X said, "..."' lines, sessions separated by a blank line), (M) Mem0-style 'Name: text' lines,
one session per blank-line block. Same rule, same c=4096."""
import sys, re, json; sys.argv=['x']
import pandas as pd, derive_config as DC
D=json.load(open("locomo/locomo10.json"))
def fmt(d, kind):
    c=d["conversation"]; ks=sorted([k for k in c if re.fullmatch(r"session_\d+",k)], key=lambda k:int(k.split("_")[1])); out=[]
    for k in ks:
        if kind=="N":
            lines=["DATE: "+c[k+"_date_time"],"CONVERSATION:"]+['%s said, "%s"'%(t["speaker"],t["text"])+(' and shared %s.'%t["blip_caption"] if t.get("blip_caption") else '') for t in c[k]]
        else:
            lines=["%s: %s"%(t["speaker"],t["text"]) for t in c[k]]
        out.append("\n".join(lines))
    return "\n\n".join(out)
for kind in ("N","M"):
    pd.DataFrame([dict(context=fmt(d,kind)) for d in D]).to_parquet("/tmp/lcm_%s.parquet"%kind)
    cov,num,tok=DC.stats("/tmp/lcm_%s.parquet"%kind,list(range(10)),4096)
    DC.REG.CELLS["x"]=("/tmp/lcm_%s.parquet"%kind,list(range(10)),0,"J",4096)
    print(kind,"atom_cov=%.4f serial=%.4f tok/row=%d"%(cov,num,tok), DC.derive("x"))
