#!/usr/bin/env python3
import os
"""A metered, hard-capped wrapper for every paid LLM call in this project.

Today's bill was not caused by a model being expensive. It was caused by three things this module makes
impossible:

  1. No meter. Every harness called the API and threw the usage object away, so the first signal that a
     run was expensive was the credit-card statement. Here, usage is read from the response on every call
     and appended to a ledger, so spend is knowable while the run is still going.
  2. No cap. A run could not stop itself. Here, a Guard refuses to make the call that would cross its
     cap and raises BudgetExceeded. The cap is reloaded from the ledger at construction, so killing and
     restarting a job does not reset it.
  3. An unbounded output channel. `rule_informed_rank_qwen.py:37` called gemini-2.5-flash with no config
     at all: no max_output_tokens, no thinking_config. gemini-2.5-flash defaults to a *dynamic* thinking
     budget (up to 24,576 tokens) and bills thinking at the output rate, which is 8.3x its input rate. A
     10-token FactConsolidation answer could therefore bill for thousands of tokens. `Guard.gemini` forces
     thinking off and a max_output_tokens, and asserts afterwards that thoughts_token_count came back 0.

Usage:

    from budget_guard import Guard
    g = Guard("fc_gpt4omini", cap_usd=5.0)
    g.preflight(n_calls=5600, in_tok=500, out_tok=10, model="gpt-4o-mini")   # refuses before spending
    text = g.openai("...prompt...", max_tokens=10)
    print(g.report())

Prices are USD per 1M tokens and are the ones this project verified on 2026-08-09.
"""
import json, os, time, threading

PRICES = {
    "qwen": (0.0, 0.0), "llama": (0.0, 0.0), "gemma": (0.0, 0.0),                      # model -> (input $/1M, output $/1M); longest prefix wins
    "gpt-4o-mini":            (0.15,  0.60),
    "gpt-4o":                 (2.50, 10.00),
    "gpt-4.1-mini":           (0.40,  1.60),
    "gpt-4.1-nano":           (0.10,  0.40),   # 2026-09-19, BEAM reader; longest-prefix rule keeps it apart from gpt-4.1-mini
    # 2026-09-05: was (0.125, 1.00), exactly half. That is the Batch API rate, not the rate
    # these runs pay. Verified on a live leg: r4_det_5m recorded $0.4859 for 2,105,890 input
    # and 222,628 output tokens; at (0.25, 2.00) that is $0.9717 -- 2.00x. Every 5-mini
    # figure written before this line changed is half the real bill, and every 5-mini
    # CAP_USD admitted twice its face value.
    "gpt-5-mini":             (0.25,  2.00),
    "gpt-5.4-mini":           (0.75,  4.50),
    "gemini-2.5-flash":       (0.30,  2.50),
    "gemini-3-flash":         (0.25,  1.50),
}

# Reasoning models bill their hidden reasoning at the OUTPUT rate and reject `temperature` and
# `max_tokens`. MemoryAgentBench hits the same wall and resolves it by passing neither for o4-mini
# (agent.py:325-329), so matching that is protocol-faithful rather than a deviation. Capping output at
# FactConsolidation's 10 tokens would be actively wrong here: reasoning consumes the budget first and the
# answer comes back empty, which is the failure we already diagnosed once on gemini-2.5-flash.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def is_reasoning_model(model):
    return any(model.startswith(p) for p in _REASONING_PREFIXES)
LEDGER_DIR = os.environ.get("COST_LEDGER_DIR",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_ledger"))


class BudgetExceeded(RuntimeError):
    pass


class Guard:
    """One Guard per run. `name` keys the ledger, so a restarted run resumes its spend, not its budget."""

    def __init__(self, name, cap_usd, warn_frac=0.5, price=None):
        self.name, self.cap = name, float(cap_usd)
        self.warn_frac, self.price = warn_frac, price
        self._lock = threading.Lock()
        os.makedirs(LEDGER_DIR, exist_ok=True)
        self.path = os.path.join(LEDGER_DIR, name + ".jsonl")
        self.spent, self.calls = self._replay()
        self._warned = self.spent >= self.cap * warn_frac
        if self.spent:
            print("[budget] %s resuming: $%.4f already spent of $%.2f cap (%d calls)"
                  % (name, self.spent, self.cap, self.calls), flush=True)

    def _replay(self):
        """Prior spend for this run name. A kill -9 mid-run must not hand the job a fresh budget."""
        s, n = 0.0, 0
        if os.path.exists(self.path):
            for line in open(self.path):
                try:
                    s += json.loads(line)["usd"]; n += 1
                except Exception:
                    continue                      # a torn final line costs one call of accuracy, not the cap
        return s, n

    # ---- accounting -------------------------------------------------------------------------------
    def rate(self, model):
        if self.price:
            return self.price
        hit = max((k for k in PRICES if model.startswith(k)), key=len, default=None)
        if hit:                              # longest prefix, so gpt-5.4-mini is not priced as gpt-5-mini
            return PRICES[hit]
        raise KeyError("no price for %r; pass price=(in,out) to Guard" % model)

    def cost(self, model, in_tok, out_tok):
        pi, po = self.rate(model)
        return in_tok / 1e6 * pi + out_tok / 1e6 * po

    def remaining(self):
        return max(0.0, self.cap - self.spent)

    def preflight(self, n_calls, in_tok, out_tok, model):
        """Refuse to start a run whose own projection does not fit. Cheaper than finding out at call 5600."""
        proj = n_calls * self.cost(model, in_tok, out_tok)
        print("[budget] %s preflight: %d calls x (%d in, %d out) on %s -> $%.2f projected, $%.2f cap"
              % (self.name, n_calls, in_tok, out_tok, model, proj, self.cap), flush=True)
        if self.spent + proj > self.cap:
            raise BudgetExceeded("projected $%.2f + spent $%.4f exceeds cap $%.2f — raise the cap "
                                 "deliberately or shrink the run" % (proj, self.spent, self.cap))
        return proj

    def _charge(self, model, in_tok, out_tok, thoughts=0, note=""):
        usd = self.cost(model, in_tok, out_tok)
        with self._lock:
            self.spent += usd; self.calls += 1
            with open(self.path, "a") as f:
                f.write(json.dumps({"t": round(time.time()), "model": model, "in": in_tok, "out": out_tok,
                                    "thoughts": thoughts, "usd": round(usd, 8), "note": note}) + "\n")
            frac = self.spent / self.cap if self.cap else 0
            if not self._warned and frac >= self.warn_frac:
                self._warned = True
                print("[budget] %s at %.0f%% of cap ($%.4f/$%.2f, %d calls)"
                      % (self.name, 100 * frac, self.spent, self.cap, self.calls), flush=True)
            if self.spent >= self.cap:
                raise BudgetExceeded("%s hit its cap: $%.4f >= $%.2f after %d calls. Ledger: %s"
                                     % (self.name, self.spent, self.cap, self.calls, self.path))
        return usd

    def _guard_before(self, model, in_est, out_est):
        """Block the call that would cross the cap instead of discovering it afterwards."""
        if self.spent + self.cost(model, in_est, out_est) > self.cap:
            raise BudgetExceeded("%s would cross its cap on the next call ($%.4f spent, $%.2f cap)"
                                 % (self.name, self.spent, self.cap))

    # ---- providers --------------------------------------------------------------------------------
    def openai(self, prompt, model="gpt-4o-mini", max_tokens=128, temperature=0.0, system=None,
               client=None, note="", seed=0):
        """temperature=0 plus an explicit seed, and the system_fingerprint is recorded.

        MemoryAgentBench never states a decoding configuration (Appendix F covers max output tokens,
        embedding models and chunk size only), so its GPT-4o-mini column was produced at the API default,
        temperature=1.0. We cannot match an unreported setting, so we fix ours to greedy, pin a seed, and
        log the fingerprint that tells us whether the backend changed under us between runs.
        """
        from openai import OpenAI
        c = client or getattr(self, "_oai", None) or OpenAI()
        self._oai = c
        self._guard_before(model, len(prompt) // 4 + 200, max_tokens)
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        if is_reasoning_model(model):
            kw = dict(model=model, messages=msgs)      # neither temperature nor a token cap, as MAB does
        else:
            kw = dict(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
            if seed is not None:
                kw["seed"] = seed
        # 2026-09-20: a local Qwen3.5 reader through ollama's OpenAI endpoint thinks by default and ignores
        # both `think:false` and the `/no_think` suffix there; `reasoning_effort` is the field it honours
        # (measured: 318 completion tokens -> 4). MemoryLACE runs Qwen3.5 with reasoning disabled, so the
        # same-backbone comparison sets REASONING_EFFORT=none.
        if os.environ.get("REASONING_EFFORT"):
            kw["reasoning_effort"] = os.environ["REASONING_EFFORT"]
        if os.environ.get("OPENAI_EXTRA_BODY"):       # vLLM: {"chat_template_kwargs": {"enable_thinking": false}}
            import json as _json
            kw["extra_body"] = _json.loads(os.environ["OPENAI_EXTRA_BODY"])
        # A 429 is the account's tokens-per-minute ceiling, not a failed experiment: at chunk 4096 one
        # question is ~41k tokens, so a 200k TPM limit allows under five calls a minute and a long run
        # will hit it. Retrying costs nothing here because _charge runs only after a response arrives,
        # so a rejected attempt is never billed and never enters the ledger.
        for attempt in range(8):
            try:
                r = c.chat.completions.create(**kw)
                break
            except Exception as e:
                if type(e).__name__ != "RateLimitError" or attempt == 7:
                    raise
                wait = min(60.0, 2.0 * (2 ** attempt))
                print("[budget] 429 rate limit; sleeping %.0fs (attempt %d/8)" % (wait, attempt + 1),
                      flush=True)
                time.sleep(wait)
        u = r.usage
        self._charge(model, u.prompt_tokens, u.completion_tokens, 0,
                     "%s|T=%s|seed=%s|fp=%s" % (note, temperature, seed,
                                                getattr(r, "system_fingerprint", None)))
        return (r.choices[0].message.content or "").strip()

    def gemini(self, prompt, model="gemini-2.5-flash", max_tokens=128, temperature=0.0,
               client=None, note="", allow_thinking=False):
        """Thinking is OFF unless explicitly allowed, and the response is checked to prove it."""
        from google import genai
        from google.genai.types import GenerateContentConfig, ThinkingConfig
        c = client or getattr(self, "_gem", None) or genai.Client()
        self._gem = c
        self._guard_before(model, len(prompt) // 4 + 200, max_tokens)
        cfg = GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens,
                                    thinking_config=ThinkingConfig(
                                        thinking_budget=-1 if allow_thinking else 0))
        r = c.models.generate_content(model=model, contents=prompt, config=cfg)
        m = r.usage_metadata
        th = getattr(m, "thoughts_token_count", 0) or 0
        if th and not allow_thinking:
            raise RuntimeError("thinking was billed despite thinking_budget=0 (%d thought tokens) — "
                               "the SDK ignored the config; stop the run" % th)
        self._charge(model, m.prompt_token_count or 0,
                     (m.candidates_token_count or 0) + th, th, note)
        return (r.text or "").strip()

    def report(self):
        return "[budget] %s: $%.4f of $%.2f cap over %d calls -> %s" % (
            self.name, self.spent, self.cap, self.calls, self.path)


def report_all():
    """`python budget_guard.py` — every run's spend, so the bill is never a surprise again."""
    if not os.path.isdir(LEDGER_DIR):
        return "no ledger at " + LEDGER_DIR
    out, grand = [], 0.0
    for fn in sorted(os.listdir(LEDGER_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        s = n = 0; th = 0
        for line in open(os.path.join(LEDGER_DIR, fn)):
            try:
                d = json.loads(line)
            except Exception:
                continue
            s += d["usd"]; n += 1; th += d.get("thoughts", 0)
        grand += s
        out.append("  %-34s $%8.4f  %6d calls  %8d thought-tokens" % (fn[:-6], s, n, th))
    return "\n".join(["cost ledger: " + LEDGER_DIR] + out + ["  %-34s $%8.4f" % ("TOTAL", grand)])


if __name__ == "__main__":
    print(report_all())
