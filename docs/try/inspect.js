import { Tiktoken } from "js-tiktoken/lite";
import cl100k from "js-tiktoken/ranks/cl100k_base";
import { decimalRanges } from "./decimal-ranges.js";
import { wordRanges } from "./word-ranges.js";

// cl100k's Rust regex uses Unicode White_Space. JavaScript's \s instead includes BOM and misses NEL.
const tokenizerPattern = cl100k.pat_str.replaceAll("\\s", "\\p{White_Space}").replaceAll("\\S", "\\P{White_Space}");
const encoder = new Tiktoken({...cl100k, pat_str: tokenizerPattern});
// Python str.strip / re \s includes controls and NEL, but excludes the BOM.
const WS = "[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const wsChar = new RegExp("^" + WS + "$", "u");
const decimalClass = decimalRanges.map(([a, b]) => `\\u{${a.toString(16)}}-\\u{${b.toString(16)}}`).join("");
const nonspace = "(?!(?:" + WS + "))[\\s\\S]";
const numberedPattern = new RegExp("^" + WS + "*[" + decimalClass + "]+\\." + WS + "+" + nonspace, "u");
const leadPattern = new RegExp("^(" + WS + "*(?:[\\[(#]" + WS + "*)?(?:[A-Za-z]+" + WS + "*)?)([" + decimalClass + "]{1,8})([\\]).:]?" + WS + "+)(?=" + nonspace + ")", "u");
// Every character of WS is one UTF-16 unit, so these walk the ends in linear time.
function pyStrip(text) {
  let start = 0, end = text.length;
  while (start < end && wsChar.test(text[start])) start += 1;
  while (end > start && wsChar.test(text[end - 1])) end -= 1;
  return text.slice(start, end);
}
const nonempty = text => pyStrip(text).length > 0;
const validUnicode = value => typeof value === "string" && !Array.from(value).some(c => c.codePointAt(0) >= 0xd800 && c.codePointAt(0) <= 0xdfff);
const sourceCharacter = "(?!(?:" + WS + "|\\[|\\]))[\\s\\S]";
const datedSpeakerPattern = new RegExp("^" + WS + "*(?:(?:" + sourceCharacter + "){1,64}" + WS + "+)?\\[([^\\]\\n]{1,128})\\]" + WS + "+([^:\\n]{1,64}):[ \\t]+" + nonspace, "u");
const digit = "[" + decimalClass + "]";
// Pin Python's word class: the host JS engine may use a newer Unicode version.
const wordClass = wordRanges.map(([a, b]) => `\\u{${a.toString(16)}}-\\u{${b.toString(16)}}`).join("");
const month = "(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\\.?";
const timePattern = new RegExp("(?:^|[^" + wordClass + "])(?:" + digit + "{4}[-/]" + digit + "{1,2}[-/]" + digit + "{1,2}|" + digit + "{1,2}:" + digit + "{2}(?::" + digit + "{2})?|" + month + "[- ]" + digit + "{1,2}(?:st|nd|rd|th)?,?[- ]" + digit + "{4}|" + digit + "{1,2}(?:st|nd|rd|th)?[- ]" + month + ",?[- ]" + digit + "{4})(?=$|[^" + wordClass + "])", "u");
const turnHeadPattern = new RegExp("^[ \\t]*(?:(?:" + sourceCharacter + "){1,64}[ \\t]+)?(?:\\[([^\\]\\n]{1,128})\\][ \\t]+)?([A-Za-z][" + wordClass + " .'-]{0,31}):[ \\t]+" + nonspace, "u");
// Python re.IGNORECASE on ASCII letters includes these Unicode equivalents.
const hasTime = text => timePattern.test(text.replace(/[A-ZİıſK]/gu, c => ({"İ":"i","ı":"i","ſ":"s","K":"k"}[c] || c.toLowerCase())));

function datedTurnShare(text) {
  const lines = text.split("\n").filter(nonempty);
  const heads = [];
  lines.forEach((line, i) => {
    const match = turnHeadPattern.exec(line);
    if (match) heads.push([i, match]);
  });
  const speakers = new Set(heads.filter(([, match]) => match[1]).map(([, match]) => pyStrip(match[2])));
  const starts = new Map(heads.filter(([, match]) => speakers.has(pyStrip(match[2]))));
  const bracketed = [...starts.values()].filter(match => match[1]).length;
  if (starts.size < 2 || bracketed < 0.1 * starts.size) return 0;
  let total = 0, inside = 0, dated = false, inTurn = false;
  lines.forEach((line, i) => {
    const size = Array.from(pyStrip(line)).length;
    total += size;
    const match = starts.get(i);
    if (match) {
      inTurn = true;
      if (match[1] && hasTime(match[1])) dated = true;
    }
    if (inTurn && dated) inside += size;
  });
  return inside / Math.max(total, 1);
}

function datedConversation(text) {
  const blocks = text.split("\n\n").filter(nonempty);
  let count = 0;
  if (blocks.length >= 2) {
    for (const block of blocks) {
      const first = block.split("\n").find(nonempty);
      const match = datedSpeakerPattern.exec(first);
      if (match && hasTime(match[1])) count += 1;
    }
  }
  return count / Math.max(blocks.length, 1) >= 0.8 || datedTurnShare(text) >= 0.8;
}

function sessionJSON(text) {
  if (!pyStrip(text).startsWith("[")) return false;
  let source;
  try { source = JSON.parse(text); } catch {
    // The Python SDK also reads Python literals. This browser accepts the JSON spelling of that input.
    if (/^\s*\[\s*['"][^'"\n]+['"]\s*,\s*\[\s*\{/.test(text)) {
      throw new Error("Use JSON for session arrays in this preview, or inspect the Python literal with muster.derive().");
    }
    return false;
  }
  const pythonJSON = value => value !== null && typeof value !== "boolean" &&
    (Array.isArray(value) ? value.every(pythonJSON) : typeof value === "object" ? Object.values(value).every(pythonJSON) : true);
  if (!Array.isArray(source) || !source.length || source.length % 2) return false;
  const spellings = sessionStringSpellings(text);
  if (!spellings || !pythonJSON(source)) return false;
  let user = false;
  for (let i = 0; i < source.length; i += 2) {
    if (!validUnicode(source[i]) || !spellings[i] || !nonempty(source[i]) || !Array.isArray(source[i + 1]) || !source[i + 1].length) return false;
    for (let j = 0; j < source[i + 1].length; j += 1) {
      const turn = source[i + 1][j];
      if (!turn || Array.isArray(turn) || typeof turn !== "object" || !["user", "assistant", "system"].includes(turn.role) || !validUnicode(turn.content) || !spellings[i + 1]?.[j]?.content) return false;
      user ||= turn.role === "user";
    }
  }
  return user;
}

// Python literal_eval leaves a \uD800/\uDC00 escape as a surrogate even when
// JSON.parse would combine a pair. Keep spelling validity for dates/content;
// unrelated source fields do not influence the SDK's session validation.
function sessionStringSpellings(text) {
  const tokens = text.match(/"(?:\\[\s\S]|[^"\\])*"|[{}\[\]:,]|[^\s{}\[\]:,]+/gu);
  let at = 0;
  function stringValid(raw) {
    for (let i = 1; i < raw.length - 1; i += 1) {
      if (raw[i] !== "\\") continue;
      if (raw[i + 1] === "u") {
        const code = parseInt(raw.slice(i + 2, i + 6), 16);
        if (code >= 0xd800 && code <= 0xdfff) return false;
        i += 5;
      } else i += 1;
    }
    return true;
  }
  function read(depth = 0) {
    const token = tokens[at++];
    if ((token === "[" || token === "{") && depth >= 200) throw new Error("literal nesting");
    if (token === "[") {
      const result = [];
      while (tokens[at] !== "]") {
        result.push(read(depth + 1));
        if (tokens[at] !== ",") break;
        at += 1;
      }
      if (tokens[at++] !== "]") throw new Error("array spelling");
      return result;
    }
    if (token === "{") {
      const result = Object.create(null);
      while (tokens[at] !== "}") {
        const key = JSON.parse(tokens[at++]);
        if (tokens[at++] !== ":") throw new Error("object spelling");
        result[key] = read(depth + 1);
        if (tokens[at] !== ",") break;
        at += 1;
      }
      if (tokens[at++] !== "}") throw new Error("object spelling");
      return result;
    }
    return token?.startsWith('"') ? stringValid(token) : null;
  }
  try { return read(); } catch { return null; }
}

function leadMatch(line) {
  const match = leadPattern.exec(line);
  return match && Array.from(match[1] + match[2]).length <= 32 ? match : null;
}

function leadingSerial(line) {
  const match = leadMatch(line);
  if (!match) return null;
  let value = 0;
  for (const character of match[2]) {
    const code = character.codePointAt(0);
    const range = decimalRanges.find(([a, b]) => a <= code && code <= b);
    if (!range) return null;
    value = value * 10 + (code - range[0]) % 10;
  }
  return value;
}

function hasShortConsecutiveSerials(lines) {
  let previous = null, steps = 0;
  for (const line of lines) {
    const value = leadingSerial(line);
    if (previous !== null && value !== null && value === previous + 1) steps += 1;
    previous = value;
  }
  if (steps / Math.max(lines.length - 1, 1) < 0.95) return false;
  const sizes = lines.map(line => encoder.encode(line, [], []).length).sort((a, b) => a - b);
  if (!sizes.length) return false;
  const middle = Math.floor(sizes.length / 2);
  const median = sizes.length % 2 ? sizes[middle] : (sizes[middle - 1] + sizes[middle]) / 2;
  return median <= 20;
}

export function inspect(text) {
  if (typeof text !== "string") throw new Error("Enter text to inspect.");
  if (text.length > 200000) throw new RangeError("This preview accepts up to 200,000 characters. Use the Python SDK for larger stores; no text was truncated.");
  if (!validUnicode(text)) throw new Error("Use valid Unicode text; an incomplete surrogate cannot be inspected.");
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
  if (hasShortConsecutiveSerials(lines)) return {...common, branch: "numbered", atom: null, chunk: null, mscale: 0, U: 10, policy: "attprio_sess", render: "numbered", supersede: "numbered", serial: "lead"};
  if (sessionJSON(normalized)) return {...common, branch: "rounds", atom: null, chunk: null, mscale: 0, U: 10, policy: "flrs", render: "flat", supersede: "off"};
  const record = recordShare >= 0.9;
  const conversation = record && datedConversation(normalized);
  const chunk = record ? 512 : 1024;
  const units = Math.floor(tokens / chunk);
  const multiplier = [8, 4, 2].find(m => Math.floor(units / m) >= 10);
  return {...common, branch: record ? "record" : "prose", atom: record ? "blank" : "sentence", chunk,
    mscale: multiplier ? chunk * multiplier : 0, U: record ? 10 : 40,
    policy: !record || conversation ? "flr" : "flrs", render: !record || conversation ? "r4" : "flat", supersede: "off"};
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = event => {
    try { self.postMessage({id: event.data.id, result: inspect(event.data.text)}); }
    catch (error) { self.postMessage({id: event.data.id, error: String(error.message || error)}); }
  };
}
