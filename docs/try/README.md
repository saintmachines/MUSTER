---
title: MUSTER Structure Inspector
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
---

# MUSTER Structure Inspector

Paste raw text or a JSON session array to inspect its structure and native retrieval configuration. The app runs in your browser with a bundled cl100k_base tokenizer. Text is not uploaded. There is no model inference or API key.

This is a configuration preview of the SDK's **raw-text single-store native** rule at reference chunk 4,096. `COMPATIBILITY.json` records the exact tested SDK sources. Independent records use `flrs/flat`, dated speaker records use `flr/r4`, and prose uses `flr/r4`. Integer-dot records and short consecutive serial logs retain their numbered rule, including `Step 12:`, `#12` and `(12)`, with the SDK's 95% consecutive-line and 20-token median checks.

Session arrays use the JSON spelling `["date", [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]]`. Supported arrays use round units under source sessions, with `flrs/flat`. Python-literal session arrays are explicitly refused by the browser; inspect those with `muster.derive()` in Python. Typed record metadata and `build_collection()` are SDK features and are not implemented in this raw-text preview. The inspector does not generate answers, measure accuracy, construct a retrieval index or certify paper replay.

Python reference: `muster.derive(your_text)`. Code: https://github.com/saintmachines/MUSTER . Paper and reproduction: https://saintmachines.github.io/MUSTER/ . The public arXiv ID will be linked when announced.

The browser implementation is checked against the pinned Python SDK; see `COMPATIBILITY.json` in the prepared delivery. Browser tokenization uses js-tiktoken 1.0.21, an MIT-licensed port (notice included). MUSTER app code: Apache-2.0. Synthetic examples illustrate configuration only.

Build on a development machine: `npm ci && npm run build`. The same files run under the project's GitHub Pages `/try/` path or a Static Space. Deploy `index.html`, `ui.js`, `inspect.bundle.js`, `README.md`, licenses and compatibility report. Static hosting requires no paid compute plan. There is no runtime CDN or external tokenizer download.

Large inputs can take longer to count locally. The page shows progress and offers **Cancel inspection**, which stops the calculation and keeps your text. Editing the text also cancels a pending inspection; select Inspect structure to calculate the new configuration. No input is shortened. The preview limit is 200,000 UTF-16 code units.
