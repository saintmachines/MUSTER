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

Paste text to see the two store statistics and the retrieval configuration they imply. The app runs in your browser with a bundled cl100k_base tokenizer. Text is not uploaded. There is no model inference or API key.

This is a configuration preview of the SDK's **single-store native** rule at reference chunk 4,096, pinned to MUSTER commit `3aed83d7c2f35bafb39c8e3cc015ec8d6e074996`. It does not generate answers, measure accuracy, construct a retrieval index, or implement collection-exact paper replay. It detects the default integer-dot line syntax; other ordered-record formats need the SDK's explicit metadata adapter.

Python reference: `muster.derive(your_text)`. Code: https://github.com/saintmachines/MUSTER . Paper and reproduction: https://saintmachines.github.io/MUSTER/ . The public arXiv ID will be linked when announced.

The browser implementation is checked against the pinned Python SDK; see `COMPATIBILITY.json` in the prepared delivery. Browser tokenization uses js-tiktoken 1.0.21, an MIT-licensed port (notice included). MUSTER app code: Apache-2.0. Synthetic examples illustrate configuration only.

Build on a development machine: `npm ci && npm run build`. The same files run under the project's GitHub Pages `/try/` path or a Static Space. Deploy `index.html`, `ui.js`, `inspect.bundle.js`, `README.md`, licenses and compatibility report. Static hosting requires no paid compute plan. There is no runtime CDN or external tokenizer download.
