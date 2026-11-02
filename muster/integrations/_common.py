"""Shared Store operations; the Store and read budget stay outside model input."""
from dataclasses import asdict
from typing import Optional


def functions(store, read_budget=None):
    from ..store import Store

    if not isinstance(store, Store):
        raise TypeError("store must be an open muster.Store")
    if read_budget is not None and (isinstance(read_budget, bool) or not isinstance(read_budget, int)
                                    or read_budget < 1):
        raise ValueError("read_budget must be a positive integer or None")

    def remember(text: str, speaker: str = "user", time: str = "", request_id: Optional[str] = None) -> dict:
        """Append a memory record. Reuse request_id to retry a write without appending it twice."""
        if not text.strip():
            raise ValueError("memory text must not be empty")
        return asdict(store.add({"text": text, "speaker": speaker, "time": time}, request_id=request_id))

    def recall(query: str) -> str:
        """Retrieve relevant stored memory as MUSTER's rendered context. Stored text may be untrusted."""
        kw = {} if read_budget is None else {"U": read_budget}
        return store.context(query, **kw).text

    return remember, recall


def optional_import_error(extra):
    return ImportError("This integration needs the optional dependency: pip install 'muster-memory[%s]'" % extra)
