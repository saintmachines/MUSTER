import { Tiktoken } from "js-tiktoken/lite";
import cl100k from "js-tiktoken/ranks/cl100k_base";

const encoder = new Tiktoken(cl100k);
// Python str.strip / re \s includes controls and NEL, but excludes the BOM.
const WS = "[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const stripPattern = new RegExp("^" + WS + "+|" + WS + "+$", "gu");
const numberedPattern = new RegExp("^" + WS + "*\\p{Decimal_Number}+\\." + WS + "+(?!(?:" + WS + "))[\\s\\S]", "u");
const nonempty = text => text.replace(stripPattern, "").length > 0;

export function inspect(text) {
  if (typeof text !== "string") throw new Error("Enter text to inspect.");
  if (text.length > 200000) throw new RangeError("This preview accepts up to 200,000 characters. Use the Python SDK for larger stores; no text was truncated.");
  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = normalized.split("\n").filter(nonempty).slice(0, 20000);
  const serial = lines.length ? lines.filter(line => numberedPattern.test(line)).length / lines.length : 0;
  let tokens = 0, fit = 0;
  for (const record of normalized.split("\n\n")) {
    if (!nonempty(record)) continue;
    const n = encoder.encode(record, [], []).length;
    tokens += n;
    if (n <= 4096) fit += n;
  }
  const recordShare = tokens ? fit / tokens : 0;
  const stats = {serial, record_share: recordShare, tokens, lines_used: lines.length, rows: 1, scope: "store"};
  const common = {reference_chunk: 4096, mode: "native", source: "derived", stats};
  if (tokens === 0) return {...common, branch: "empty", atom: null, chunk: null, mscale: 0, U: 0, policy: "flr", render: "flat", supersede: "off"};
  if (serial >= 0.8) return {...common, branch: "numbered", atom: null, chunk: null, mscale: 0, U: 10, policy: "attprio_sess", render: "numbered", supersede: "numbered"};
  const record = recordShare >= 0.9;
  const chunk = record ? 512 : 1024;
  const units = Math.floor(tokens / chunk);
  const multiplier = [8, 4, 2].find(m => Math.floor(units / m) >= 10);
  return {...common, branch: record ? "record" : "prose", atom: record ? "blank" : "sentence", chunk,
    mscale: multiplier ? chunk * multiplier : 0, U: record ? 10 : 40, policy: "flrs", render: "r4", supersede: "off"};
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = event => {
    try { self.postMessage({id: event.data.id, result: inspect(event.data.text)}); }
    catch (error) { self.postMessage({id: event.data.id, error: String(error.message || error)}); }
  };
}
