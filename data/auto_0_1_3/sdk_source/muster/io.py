"""Reading a store from a file: .txt, .jsonl (a column, records joined by a blank line) or .parquet (a column, a row)."""
import json

from .errors import MusterError


def read_store(path: str, column: str = "context", row: int = 0) -> str:
    if path.endswith(".parquet"):
        try:
            import pandas as pd
        except ImportError:
            raise MusterError("reading .parquet needs pandas and pyarrow: pip install 'muster-memory[parquet]'")
        df = pd.read_parquet(path)
        if column not in df.columns:
            raise MusterError("%s has no column %r" % (path, column))
        if not 0 <= row < len(df):
            raise MusterError("%s has %d rows; row %d is out of range" % (path, len(df), row))
        return str(df[column].iloc[row])
    if path.endswith(".jsonl"):
        parts = []
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    parts.append(str(json.loads(line)[column]))
                except json.JSONDecodeError:
                    raise MusterError("%s line %d is not JSON" % (path, n))
                except (KeyError, TypeError):
                    raise MusterError("%s line %d has no field %r" % (path, n, column))
        return "\n\n".join(parts)
    with open(path, encoding="utf-8") as f:
        return f.read()
