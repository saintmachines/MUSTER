"""Text-keyed caches: reranker scores and embedding vectors, in one SQLite file or in memory.

A score is keyed by (reranker, query, unit text) and a vector by (encoder, key text), so entries stay valid when a
store grows and its unit indices move.
"""
import hashlib
import sqlite3
import threading


def _sha(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).digest()


def model_key(obj):
    return "%s@%s" % (getattr(obj, "name", type(obj).__name__), getattr(obj, "revision", "") or "")


class ScoreCache:
    """Reranker scores. `path=None` keeps them in memory."""

    def __init__(self, path=None, *, model="reranker"):
        self.model = model
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores(model TEXT, qsha BLOB, usha BLOB, score REAL, "
                        "PRIMARY KEY (model, qsha, usha))")
        self.db.commit()

    def get_many(self, query, texts):
        q = _sha(query)
        out = {}
        with self._lock:
            for t in set(texts):
                r = self.db.execute("SELECT score FROM scores WHERE model=? AND qsha=? AND usha=?",
                                    (self.model, q, _sha(t))).fetchone()
                if r is not None:
                    out[t] = r[0]
        return out

    def put_many(self, query, scores):
        q = _sha(query)
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
                                [(self.model, q, _sha(t), float(v)) for t, v in scores.items()])
            self.db.commit()


class VectorCache:
    """Embedding vectors (float32) keyed by key text."""

    def __init__(self, db, model, *, create=True):
        self.db, self.model = db, model
        if create:
            self.db.execute("CREATE TABLE IF NOT EXISTS vectors(model TEXT, ksha BLOB, dim INT, vec BLOB, "
                            "PRIMARY KEY (model, ksha))")

    def get_many(self, keys):
        out = {}
        for k in set(keys):
            r = self.db.execute("SELECT vec FROM vectors WHERE model=? AND ksha=?", (self.model, _sha(k))).fetchone()
            if r is not None:
                out[k] = r[0]
        return out

    def put_many(self, items):
        self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                            [(self.model, _sha(k), len(v) // 4, v) for k, v in items.items()])
