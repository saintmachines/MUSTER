"""The ONLY way this project calls OpenRouter chat models.

Rules applied to every OpenRouter call:
  * every model is pinned to ONE provider that does not train on prompts (data policies read from OpenRouter's provider
    list on 2026-09-22), sent as provider.order=[slug] with allow_fallbacks=False and require_parameters=True, so a request
    can never be re-routed to an fp4 copy or a training endpoint;
  * the provider that actually answered is checked against the pin on every response -- a mismatch raises;
  * account settings (openrouter.ai/settings/privacy): ZDR off, paid-training off, free-training off, free-publish off,
    1% discount off (the second, account-wide guard: training endpoints are refused even if a pin were wrong);
  * the key lives only in the environment (`source .keys.sh`, written with `read -rs`, or Colab Secrets), never in code.
Models outside PINS are refused. PINS[m]['off'] = the lowest reasoning setting that endpoint accepts
(smoke-tested 2026-09-22: every pin answered from its pinned provider; GLM on Phala rejects effort=none; Gemini thinking cannot be off). Add a model only after reading its endpoints (quantization) and the provider's data policy.
"""
import json, os, threading, time

PINS = {
    # 2026-10-08 (author: Hindsight write-time LLM = gpt-oss-120b as in the Hindsight paper): endpoints read 2026-10-08; two DeepInfra rows
    # (deepinfra/bf16 $0.037/$0.17 maxout 117964; deepinfra/turbo labelled bf16 $0.15/$0.60 maxout 16384) -> pinned to the exact tag deepinfra/bf16.
    "openai/gpt-oss-120b":           dict(slug="deepinfra/bf16", names={"DeepInfra"},                    quant="bf16", train=False, retain=False, seed=True, off={"effort": "low"}),
    # Haiku 5.5 as a cross-vendor LoCoMo judge: endpoints read 2026-10-08, ten providers; pinned to Anthropic
    # first-party (commercial terms: no training on API content; standard retention). No temperature or seed parameter on any endpoint.
    "anthropic/claude-haiku-5.5":    dict(slug="anthropic",     names={"Anthropic"},                    quant="n/a", train=False, retain="30d", seed=False, off={"effort": "none"}, no_temperature=True),
    # model id                         provider slug     names OpenRouter reports   quant   train  retain   seed
    "z-ai/glm-5.3-flash":            dict(slug="phala",         names={"Phala"},                        quant="fp8", train=False, retain=False, seed=True, off={"effort": "minimal"}),   # reasoning mandatory on Phala
    "xiaomi/mimo-v2.6-pro":          dict(slug="xiaomi",        names={"Xiaomi"},                       quant="fp8", train=False, retain="30d", seed=False, off={"effort": "none"}),
    "deepseek/deepseek-v4.1-flash":  dict(slug="deepinfra",     names={"DeepInfra"},                    quant="fp8", train=False, retain=False, seed=True, off={"effort": "none"}),
    "qwen/qwen3.8-flash":            dict(slug="alibaba",       names={"Alibaba", "Alibaba Cloud Int."}, quant="unstated", train=False, retain=True, seed=True, off={"effort": "none"}),
    "google/gemini-3.1-pro-preview": dict(slug="google-vertex", names={"Google", "Google Vertex"},     quant="n/a", train=False, retain=False, seed=True, off={"effort": "low"}),   # thinking cannot be disabled
    # readers for the Mem0-context reader swap (2026-09-23; OpenRouter endpoints read 2026-09-23: openai/gpt-4.1-mini -> OpenAI,
    # Azure; qwen/qwen3.5-9b -> DeepInfra bf16 among six). Non-reasoning / thinking off, matching the earlier MUSTER rows
    # (gpt-4.1-mini harness reader; Qwen3.5-9B Colab reader with enable_thinking False, temperature 0, seed 42, 1,200 tokens).
    "openai/gpt-4.1-mini":           dict(slug="openai",        names={"OpenAI"},                       quant="n/a", train=False, retain="30d", seed=True, off={"effort": "none"}, no_reasoning_param=True),
    # gpt-5-mini (2026-09-25, author: BEAM reader matching the MAB board reader): OpenAI endpoint, reasoning at the API default (medium)
    "openai/gpt-5-mini":             dict(slug="openai",        names={"OpenAI"},                       quant="n/a", train=False, retain="30d", seed=True, off={"effort": "minimal"}, no_temperature=True),
    # 2026-09-25 17:10 (the author chose Parasail): DeepInfra returned 429 engine_overloaded for ~3 h (uptime null); Parasail
    # serves the same qwen3.5-9b-20260310 at bf16. 100K C a20 and Mem0 are re-read on Parasail as a provider check. Old pin: deepinfra/bf16.
    "qwen/qwen3.5-9b":               dict(slug="parasail/bf16",  names={"Parasail"},                    quant="bf16", train=False, retain=False, seed=True, off={"effort": "none"}),
    # judges used by competitors (2026-09-22 audit: scratchpad judge_audit/or, verified by an independent fetch):
    "z-ai/glm-5.2":                  dict(slug="z-ai",          names={"Z.AI"},                         quant="fp8", train=False, retain=False, seed=False, off=None),   # ValorBrain's judge; efforts high/xhigh only, seed not accepted
    "google/gemini-2.5-flash-lite":  dict(slug="google-vertex/eu", names={"Google", "Google Vertex"},   quant="n/a", train=False, retain=False, seed=True, off=None),    # Hindsight/AMB 5d5e8db judge; EU region (global endpoint degraded)
}
BASE_URL = "https://openrouter.ai/api/v1"
_lock = threading.Lock()
_client = None


class PinViolation(RuntimeError):
    pass


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not in the environment")
        _client = OpenAI(base_url=BASE_URL, api_key=key)
    return _client


def chat(model, messages, *, reasoning=None, temperature=0.0, max_tokens=None, seed=42, ledger=None, note="",
         spent=None, cap=None, retries=6, client=None, response_format=None):
    """One chat completion through the pinned provider. Returns (text, info) where info carries provider, usage and cost.
    reasoning: OpenRouter's reasoning object, e.g. {"effort": "low"} (Gemini 3.x: low/medium/high -> thinkingLevel) or
    {"effort": "none"} where the model allows turning it off. spent/cap: a dict {'usd': x} and a $ ceiling checked BEFORE the call."""
    if model not in PINS:
        raise PinViolation(f"{model} is not pinned in or_client.PINS -- read its endpoints and provider data policy first")
    pin = PINS[model]
    if spent is not None and cap is not None:
        with _lock:
            if spent["usd"] >= cap:
                raise RuntimeError(f"cap ${cap:.2f} reached -- no call made")
    extra = {"provider": {"order": [pin["slug"]], "allow_fallbacks": False, "require_parameters": True},
             "usage": {"include": True}}
    # no_reasoning_param: a non-reasoning model (gpt-4.1-mini) has no endpoint that accepts `reasoning` under
    # require_parameters=True, so the parameter is omitted; the pin still fixes the provider.
    if reasoning is not None and not pin.get("no_reasoning_param"):
        extra["reasoning"] = reasoning
    kw = dict(model=model, messages=messages, extra_body=extra)
    if temperature is not None and not pin.get("no_temperature"):   # gpt-5-mini accepts no temperature; with require_parameters it would match no endpoint
        kw["temperature"] = temperature
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    if response_format is not None:   # structured output (json_schema), e.g. to mirror a Gemini response_schema judge
        kw["response_format"] = response_format
    if seed is not None and pin["seed"]:
        kw["seed"] = seed
    c = client or _get_client()
    for attempt in range(retries):
        try:
            r = c.chat.completions.create(**kw)
            if r.choices and r.choices[0].finish_reason == "error":
                # upstream provider error delivered as a 200 with no answer (seen 13/400 on Gemini 3.1 Pro, 2026-09-22:
                # 0 completion tokens, $0) -- a failed call, not an answer, so it is retried like any other failure
                raise RuntimeError(f"{model}: finish_reason=error from the provider (no answer)")
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt))
    extra_fields = getattr(r, "model_extra", None) or {}
    provider = getattr(r, "provider", None) or extra_fields.get("provider")
    if provider not in pin["names"]:
        raise PinViolation(f"{model}: answered by {provider!r}, pinned to {pin['slug']} ({sorted(pin['names'])})")
    u = r.usage
    ud = u.model_dump() if hasattr(u, "model_dump") else dict(u)
    cost = float(ud.get("cost") or 0.0)
    rt = ((ud.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
    info = dict(model=model, provider=provider, prompt_tokens=ud.get("prompt_tokens"), completion_tokens=ud.get("completion_tokens"),
                reasoning_tokens=rt, cost=cost, finish_reason=r.choices[0].finish_reason)   # 'length' = max_tokens cut the output
    with _lock:
        if spent is not None:
            spent["usd"] += cost
        if ledger:
            with open(ledger, "a") as f:
                f.write(json.dumps(dict(t=int(time.time()), note=note, usd=round(cost, 7), **info)) + "\n")
    msg = r.choices[0].message
    info["reasoning_text"] = (getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning") or "")
    return (msg.content or "").strip(), info
