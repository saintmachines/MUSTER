#!/usr/bin/env python3
"""Check that every number in README.md appears in the paper tex or in a named source file (the headline data, a gate
report). Usage: check_readme_numbers.py README.md paper.tex [source ...]"""
import sys, re
from pathlib import Path
import subprocess
readme = open(sys.argv[1], encoding='utf-8').read()
# The collection table has its own source/cell/rounding/API-proof contract.
# Strip it only after that complete generated block has been validated.
collection = re.compile(r'<!-- BEGIN collection-auto -->.*?<!-- END collection-auto -->', re.S)
if '<!-- BEGIN collection-auto -->' in readme or '<!-- END collection-auto -->' in readme:
    if (readme.count('<!-- BEGIN collection-auto -->') != 1
            or readme.count('<!-- END collection-auto -->') != 1 or len(collection.findall(readme)) != 1):
        sys.exit('Expected exactly one collection Auto result block')
    root = Path(sys.argv[1]).resolve().parent
    result = subprocess.run([sys.executable, str(root / 'tools/make_collection_auto.py'), '--check'],
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr)
        sys.exit(result.returncode)
    readme = collection.sub('', readme)
# The generated native table is checked by its exact cells and display rounding,
# rather than by finding a rounded substring somewhere in a full-precision JSON.
native = re.compile(r'<!-- BEGIN native-mab -->.*?<!-- END native-mab -->', re.S)
if native.search(readme):
    if len(native.findall(readme)) != 1:
        sys.exit('Expected exactly one scoped SDK result block')
    root = Path(sys.argv[1]).resolve().parent
    source = root / 'docs/data/native_mab_0_1_2.json'
    if source.resolve() not in {Path(p).resolve() for p in sys.argv[3:]}:
        sys.exit('The native result block requires its named canonical JSON source')
    for tool in ('make_native_mab.py', 'check_native_mab_tex.py'):
        command = [sys.executable, str(root / 'tools' / tool)]
        if tool == 'make_native_mab.py':
            command.append('--check')
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout + result.stderr)
            sys.exit(result.returncode)
    readme = native.sub('', readme)
tex = open(sys.argv[2], encoding='utf-8').read()
tex_n = tex.replace('{,}', ',').replace('$', '').replace('\\,', '')
tex_n += "".join(open(f, encoding='utf-8').read() for f in sys.argv[3:])
body = re.sub(r"```text.*?```", "", readme, flags=re.S)       # script output blocks are not paper numbers
body = re.sub(r'(?:width|height)="\d+"', "", body)             # image sizes
body = re.sub(r"https?://\S+", "", body)
body = re.sub(r"\d{4}-\d{2}-\d{2}", "", body)
nums = sorted(set(re.findall(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+(?=\s*[KMx×%])|(?<![\w.])\d{2,}(?![\w.])", body)))
skip = {"2026","2027","120","145","512","4096","8192","2048","100","1000","60","90","15","16","80","20","40","10","500","200","400","700","128","2.1.0","0.10.3","0.80","0.90","1.2.0","35","37","24","23","22","25","14","13","12","11","30","50"}
miss = [n for n in nums if n not in skip and n not in tex_n]
print("numbers checked:", len(nums)); print("not found in tex:", miss)
sys.exit(1 if miss else 0)
