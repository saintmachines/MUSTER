"""Every error the library raises is a MusterError, so a caller can catch one class."""


class MusterError(Exception):
    """Base class."""


class PlanError(MusterError, ValueError):
    """A plan that cannot be built or does not fit the store it was given."""


class RecordError(MusterError, ValueError):
    """A record the library cannot read (missing text, ambiguous time, multi-line numbered text)."""


class QueryError(MusterError, ValueError):
    """A query the library cannot read: empty, not text, or not valid UTF-8."""


class StoreRefused(MusterError):
    """Exact mode refuses a store or a step that the paper's harness refuses or would crash on.

    The message names the harness line, so the refusal can be checked against the code that produced the paper."""


class EnvironmentRefused(MusterError):
    """Exact mode needs the gate machine's environment (hash seed off, pinned versions, pinned models)."""


class AssetMissing(MusterError):
    """A tokenizer or sentence-splitter file is missing or does not match its pinned hash. Run `muster fetch`."""


class StaleSelection(MusterError):
    """A selection made on one version of a memory was rendered on another."""


class PlanViolation(MusterError):
    """An add would break a frozen exact plan; nothing was changed."""


class StoreLocked(MusterError):
    """Another process holds the store's writer lock."""
