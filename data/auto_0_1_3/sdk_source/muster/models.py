"""The two models search uses: an embedding model (bge-m3) and a cross-encoder reranker (bge-reranker-v2-m3).

Neither is called to build or grow a memory's text structure; the encoder embeds each unit once, and the reranker
scores (query, unit) pairs at search time. Any object with the same `encode` / `score` methods can replace either
(a hosted embedding API, or the fakes below for tests).
"""
import hashlib
import struct

from .plan import ENCODER, RERANKER


def _device(device):
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    def __init__(self, spec=ENCODER, device="auto", cache_dir=None, local_files_only=False):
        from sentence_transformers import SentenceTransformer
        kw = dict(device=_device(device), cache_folder=cache_dir, local_files_only=local_files_only)
        if spec.revision:
            kw["revision"] = spec.revision
        self.m = SentenceTransformer(spec.name, **kw)
        self.m.max_seq_length = spec.max_length
        self.spec, self.name, self.revision, self.device = spec, spec.name, spec.revision, kw["device"]

    def encode(self, texts, batch_size=None, normalize_embeddings=True):
        import numpy as np
        v = self.m.encode(list(texts), batch_size=batch_size or self.spec.batch_size,
                          normalize_embeddings=normalize_embeddings, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class Reranker:
    def __init__(self, spec=RERANKER, device="auto", cache_dir=None, local_files_only=False):
        from sentence_transformers import CrossEncoder
        kw = dict(device=_device(device), max_length=spec.max_length, cache_folder=cache_dir,
                  local_files_only=local_files_only)
        if spec.revision:
            kw["revision"] = spec.revision
        self.m = CrossEncoder(spec.name, **kw)
        self.spec, self.name, self.revision, self.device = spec, spec.name, spec.revision, kw["device"]

    def score(self, query, docs):
        if not docs:
            return []
        return [float(v) for v in self.m.predict([(query, d) for d in docs], batch_size=self.spec.batch_size,
                                                 show_progress_bar=False)]


class Models:
    """Both models, loaded on first use. `encoder=` / `reranker=` inject other implementations."""

    def __init__(self, device="auto", *, encoder=None, reranker=None, encoder_spec=ENCODER, reranker_spec=RERANKER,
                 cache_dir=None, local_files_only=False):
        self.device, self._enc, self._rr = device, encoder, reranker
        self.encoder_spec, self.reranker_spec = encoder_spec, reranker_spec
        self.cache_dir, self.local_files_only = cache_dir, local_files_only

    @property
    def encoder(self):
        if self._enc is None:
            self._enc = Encoder(self.encoder_spec, self.device, self.cache_dir, self.local_files_only)
        return self._enc

    @property
    def reranker(self):
        if self._rr is None:
            self._rr = Reranker(self.reranker_spec, self.device, self.cache_dir, self.local_files_only)
        return self._rr

    @property
    def injected(self):
        return not (self._enc is None or isinstance(self._enc, Encoder)) or \
            not (self._rr is None or isinstance(self._rr, Reranker))

    @staticmethod
    def fake(dim=16, *, quantize=None):
        return Models(encoder=FakeEncoder(dim, quantize), reranker=FakeReranker(quantize))


def _hfloat(*parts):
    h = hashlib.sha256("\x00".join(parts).encode("utf-8", "replace")).digest()
    return struct.unpack(">Q", h[:8])[0] / 2.0 ** 64


class FakeEncoder:
    """Deterministic hash vectors of the bag of lower-cased words (tests; no weights)."""
    name, revision = "fake-encoder", "0"

    def __init__(self, dim=16, quantize=None):
        self.dim, self.quantize = dim, quantize

    def encode(self, texts, batch_size=None, normalize_embeddings=True):
        import numpy as np
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in t.lower().split():
                out[r, int(_hfloat("d", w) * self.dim)] += 1.0
            if self.quantize:
                out[r] = np.round(out[r] * self.quantize) / self.quantize
            n = float(np.linalg.norm(out[r]))
            if normalize_embeddings and n > 0:
                out[r] /= n
        return out


class FakeReranker:
    """Word overlap plus a hash jitter; `quantize` forces ties."""
    name, revision = "fake-reranker", "0"

    def __init__(self, quantize=None):
        self.quantize = quantize

    def score(self, query, docs):
        q = set(query.lower().split())
        out = []
        for d in docs:
            s = len(q & set(d.lower().split())) + _hfloat("r", query, d)
            out.append(float(round(s * self.quantize) / self.quantize) if self.quantize else s)
        return out
