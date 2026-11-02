#!/usr/bin/env python3
"""
UNIFORM RERANKER ADAPTER — one `score(query, docs) -> list[float]` over three different architectures.

Wrapping all three in sentence-transformers' CrossEncoder failed identically:
    ValueError: Cannot handle batch sizes > 1 if no padding token is defined.
because only bge-reranker-v2-m3 is an encoder with a classification head. The other two are
causal-LM-based and each has its own contract:

  bge-reranker-v2-m3   XLM-R-large encoder + classifier head. CrossEncoder is correct here.
  Qwen3-Reranker-0.6B  a causal LM asked a yes/no question. The score is the softmax over the "yes"/"no"
                       logits at the last position, using the model's own chat template — scoring it as
                       a sequence classifier reads an untrained head and returns noise even once padding
                       is fixed, so this must be implemented, not patched.
  jina-reranker-v3     LISTWISE: query and ALL candidates go into one context window and are ranked
                       against each other. That is a different objective from pointwise scoring, and it
                       is the reason to test it at all — pointwise relevance is the wrong criterion for
                       aggregation questions, where the least individually-relevant occurrence can carry
                       the highest marginal value.

Every adapter returns higher-is-better floats aligned with `docs`, so the sweep code is unchanged.
"""
import os
import torch


class BGEReranker:
    def __init__(self, name="BAAI/bge-reranker-v2-m3", device="mps", max_length=None):
        if max_length is None:
            # DEFAULT RAISED 320 -> 1024 (2026-08-19). 320 truncated EVERY unit: 512-token chunks are
            # 442-630 tokens under the reranker's XLM-R tokenizer, so a 320 cap showed the CE ~56%
            # of each one. Eight eventqa row-2 runs shipped without CEML and silently ran at 320;
            # they average 0.615 (min 0.470) against 0.700 for the six that passed CEML=1024.
            # A standard that must be re-typed in every launch script is not a standard.
            max_length = int(os.environ.get("CEML", "1024"))
        from sentence_transformers import CrossEncoder
        self.max_length = max_length   # score() needs it; it is NOT in that scope
        self.m = CrossEncoder(name, device=device, max_length=max_length)
        self.name = name

    def score(self, query, docs, batch_size=None):
        if not docs:
            return []
        if batch_size is None:
            batch_size = int(os.environ.get("CEBS", "64"))
        if os.environ.get("CE_SUBMAX", "") not in ("", "0"):
            # Split only where the model would OTHERWISE TRUNCATE. CE_SUBMAX exists to stop
            # truncation, not to shred units that already fit. Defaulting the piece size to
            # max_length makes it a no-op when the unit fits and a repair exactly when it does
            # not. The old 512 split a 569-token unit into 512 + a 57-token ORPHAN scored alone
            # that could win the max; and the splitter strides by the piece size, so the pieces
            # do not overlap and evidence on the boundary is in neither piece whole.
            st = int(os.environ.get("CE_SUB_TOK", str(self.max_length)))
            tk = self.m.tokenizer
            pieces, owner = [], []
            for i, d in enumerate(docs):
                ids = tk(d, add_special_tokens=False)["input_ids"]
                if len(ids) <= st:
                    pieces.append(d)
                    owner.append(i)
                elif os.environ.get("CE_SLIDE", "") not in ("", "0"):
                    # SLIDING, orphan-free. stride = st//2 so every position is whole inside some
                    # piece; the last piece is back-anchored to the end so its length is always st.
                    # The tiling branch below is kept as-is because published numbers ran it.
                    stride = max(1, st // 2)
                    pos = 0
                    while True:
                        if pos + st >= len(ids):
                            pieces.append(tk.decode(ids[max(0, len(ids) - st):]))
                            owner.append(i)
                            break
                        pieces.append(tk.decode(ids[pos:pos + st]))
                        owner.append(i)
                        pos += stride
                else:
                    for s in range(0, len(ids), st):
                        pieces.append(tk.decode(ids[s:s + st]))
                        owner.append(i)
            sub = self.m.predict([(query, p) for p in pieces],
                                 batch_size=batch_size, show_progress_bar=False)
            best = [-1e30] * len(docs)
            for o, v in zip(owner, sub):
                if float(v) > best[o]:
                    best[o] = float(v)
            out = best
        else:
            out = [float(v) for v in self.m.predict([(query, d) for d in docs],
                                                    batch_size=batch_size, show_progress_bar=False)]
        if os.environ.get("CEFREE", "") not in ("", "0"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return out


class QwenReranker:
    """Official yes/no-logit protocol. Anything else reads an untrained head."""
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
              "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
              "<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    TASK = "Given a user question about their long-term chat memory, retrieve passages that answer it"

    def __init__(self, name="Qwen/Qwen3-Reranker-0.6B", device="mps", max_length=512):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
        self.dev, self.max_length, self.name = device, max_length, name
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.yes_id = self.tok.convert_tokens_to_ids("yes")

    @torch.no_grad()
    def score(self, query, docs, batch_size=16):
        if not docs:
            return []
        out = []
        for a in range(0, len(docs), batch_size):
            texts = [self.PREFIX + "<Instruct>: %s\n<Query>: %s\n<Document>: %s" % (self.TASK, query, d)
                     + self.SUFFIX for d in docs[a:a + batch_size]]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length).to(self.dev)
            lg = self.m(**enc).logits[:, -1, :]
            pair = torch.stack([lg[:, self.no_id], lg[:, self.yes_id]], dim=1).float()
            out += torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return out


class JinaListwise:
    """Listwise: one forward over query + all candidates, ranked against each other."""

    def __init__(self, name="jinaai/jina-reranker-v3", device="mps", max_length=320):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name, trust_remote_code=True,
                                           dtype=torch.float16 if device != "cpu" else torch.float32)
        self.m.to(device).eval()
        self.dev, self.name, self.max_length = device, name, max_length

    @torch.no_grad()
    def score(self, query, docs, batch_size=None):
        if not docs:
            return []
        res = self.m.rerank(query, docs)   # listwise call; the model owns its own length policy
        sc = [0.0] * len(docs)
        for r in res:
            # the API returns dicts with an index and a relevance score under one of a few key names
            i = r.get("index", r.get("corpus_id"))
            v = r.get("relevance_score", r.get("score", 0.0))
            if i is not None and 0 <= int(i) < len(docs):
                sc[int(i)] = float(v)
        return sc


def build(name, device=None):
    device = device or os.environ.get("BGE_DEVICE", "mps")
    if "Qwen3-Reranker" in name:
        return QwenReranker(name, device=device)
    if "jina-reranker" in name:
        return JinaListwise(name, device=device)
    return BGEReranker(name, device=device)
