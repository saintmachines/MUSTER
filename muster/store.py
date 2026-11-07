"""Store: a growing memory in one SQLite file, for services, agents and robots.

    store = muster.Store.open("agent.muster")
    store.add([{"speaker": "user", "text": "I moved to Lisbon in May.", "time": "2026-05-02"}])
    print(store.context("Where does the user live now?").text)

`add` only appends records (one transaction, no model call), so it is fast and safe to call from a hot path. The
first search after an add builds the new version: the grown store is cut, indexed and planned exactly as a fresh
build of all its records would be, and only keys never embedded before are sent to the encoder. Reranker scores and
vectors are cached by text, so they survive restarts and growth.
"""
import json
import os
import sqlite3
import threading
import time
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Optional

from .cache import ScoreCache, VectorCache, model_key
from .errors import RecordError, StoreLocked
from .memory import Memory, build
from .records import Record, as_record, record_list, require_text


@dataclass(frozen=True)
class AddResult:
    version: int
    duplicate: bool
    n_records: int


def _rec_json(r: Record):
    d = {}
    for f in fields(Record):
        v = getattr(r, f.name)
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if f.name == "text" or v != default:
            d[f.name] = dict(v) if f.name == "meta" else (list(v) if isinstance(v, tuple) else v)
    try:
        out = json.dumps(d, ensure_ascii=False, sort_keys=True)
        out.encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecordError("a record's fields must be valid UTF-8 JSON values to be stored: %s" % exc) from exc
    return out


class Store:
    def __init__(self, path, *, models=None, plan=None, template=None, readonly=False):
        self.path, self.models, self.plan, self.template, self.readonly = os.fsdecode(path), models, plan, template, readonly
        self._lock = threading.RLock()
        # URI escaping preserves literal '?', '#' and '%' in application paths.
        # SQLite enforces read-only access and refuses to create a missing file.
        target = Path(self.path).absolute().as_uri() + "?mode=ro" if readonly else self.path
        self.db = sqlite3.connect(target, uri=readonly, check_same_thread=False, isolation_level=None)
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS records(seq INTEGER PRIMARY KEY, request_id TEXT, record TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, version INT)")
        self._head: Optional[Memory] = None
        self._head_version = -1
        self._scores = None
        self._vectors = None
        if not readonly:
            pid = self.db.execute("SELECT v FROM meta WHERE k='writer'").fetchone()
            if pid and pid[0] != str(os.getpid()) and _alive(int(pid[0])):
                raise StoreLocked("%s is open for writing by process %s" % (path, pid[0]))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('writer', ?)", (str(os.getpid()),))

    @classmethod
    def open(cls, path, **kw):
        return cls(path, **kw)

    # ------------------------------------------------------------- writing
    def add(self, records, *, request_id: Optional[str] = None) -> AddResult:
        """Append records. A repeated request_id is acknowledged once and appends nothing."""
        if self.readonly:
            raise PermissionError("store opened readonly")
        if isinstance(request_id, str):
            require_text(request_id, "request_id")
        recs = [as_record(r) for r in record_list(records)]
        with self._lock:
            cur = self.db.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                if request_id is not None:
                    r = cur.execute("SELECT version FROM requests WHERE request_id=?", (request_id,)).fetchone()
                    if r is not None:
                        cur.execute("COMMIT")
                        return AddResult(r[0], True, 0)
                cur.executemany("INSERT INTO records(request_id, record) VALUES (?, ?)",
                                [(request_id, _rec_json(x)) for x in recs])
                v = cur.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                if request_id is not None:
                    cur.execute("INSERT INTO requests VALUES (?, ?)", (request_id, v))
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise
        return AddResult(v, False, len(recs))

    # ------------------------------------------------------------- reading
    @property
    def version(self):
        return self.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def records(self):
        return [as_record(json.loads(r[0])) for r in self.db.execute("SELECT record FROM records ORDER BY seq")]

    def _models(self):
        if self.models is None:
            from .models import Models
            self.models = Models()
        return self.models

    def head(self) -> Memory:
        """The current version as a Memory, rebuilt only when records were added since the last call."""
        with self._lock:
            v = self.version
            if self._head is not None and self._head_version == v:
                return self._head
            kw = {"template": self.template} if self.template else {}
            mem = build(self.records(), self.plan, models=self._models(), embed=False, **kw)
            mem.version = v
            if len(mem):
                self._embed(mem)
            self._head, self._head_version = mem, v
            return mem

    def _embed(self, mem):
        import numpy as np
        m = self._models()
        if self._vectors is None:
            if self.readonly and not self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'").fetchone():
                self.db.execute("CREATE TEMP TABLE vectors(model TEXT, ksha BLOB, dim INT, vec BLOB, "
                                "PRIMARY KEY (model, ksha))")
            self._vectors = VectorCache(self.db, model_key(m.encoder) + "|%d" % mem.plan.encoder.max_length,
                                        create=not self.readonly)
        keys = list(mem.cut.keys)
        have = self._vectors.get_many(keys)
        todo = [k for k in dict.fromkeys(keys) if k not in have]
        if todo:
            vec = m.encoder.encode(todo, batch_size=mem.plan.encoder.batch_size, normalize_embeddings=True)
            new = {k: np.asarray(v, dtype=np.float32).tobytes() for k, v in zip(todo, vec)}
            if not self.readonly:              # a reader keeps new vectors in memory and writes nothing
                self.db.execute("BEGIN")
                self._vectors.put_many(new)
                self.db.execute("COMMIT")
            have.update(new)
        mem.vecs = np.stack([np.frombuffer(have[k], dtype=np.float32) for k in keys])

    def score_cache(self):
        if self._scores is None:
            self._scores = ScoreCache(None if (self.readonly or self.path == ":memory:") else self.path + ".scores",
                                      model=model_key(self._models().reranker))
        return self._scores

    def search(self, query, **kw):
        mem = self.head()
        return mem.search(query, scores=self.score_cache(), models=self._models(), **kw)

    def context(self, query, **kw):
        mem = self.head()
        style = kw.pop("style", None)
        return mem.render(mem.search(query, scores=self.score_cache(), models=self._models(), **kw), style=style)

    def close(self):
        if not self.readonly:
            try:
                self.db.execute("DELETE FROM meta WHERE k='writer' AND v=?", (str(os.getpid()),))
            except sqlite3.Error:
                pass
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
