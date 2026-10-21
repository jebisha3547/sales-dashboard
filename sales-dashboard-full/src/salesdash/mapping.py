"""Column mapping for uploaded files: suggest a mapping, let the user override, apply it."""
from __future__ import annotations

import pandas as pd

from .cleaning import (COLUMN_ALIASES, OPTIONAL_NUMERIC, OPTIONAL_TEXT, REQUIRED_COLUMNS,
                       MissingColumnsError, normalise_name)

TARGET_FIELDS = REQUIRED_COLUMNS + [
    c for c in OPTIONAL_TEXT + OPTIONAL_NUMERIC + ["ship_date"] if c not in REQUIRED_COLUMNS
]


def suggest_mapping(columns) -> dict[str, str | None]:
    """Best-guess {project field: source column or None} from the file's headers."""
    by_name: dict[str, str] = {}
    for col in columns:
        n = normalise_name(col)
        by_name.setdefault(COLUMN_ALIASES.get(n, n), col)
    return {field: by_name.get(field) for field in TARGET_FIELDS}


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Return a copy of df holding only the mapped columns, renamed to project fields."""
    missing = [f for f in REQUIRED_COLUMNS if not mapping.get(f)]
    if missing:
        raise MissingColumnsError(missing)
    chosen = {field: src for field, src in mapping.items() if src}
    sources = list(chosen.values())
    dupes = sorted({s for s in sources if sources.count(s) > 1})
    if dupes:
        raise ValueError(f"Source column(s) mapped to more than one field: {', '.join(map(str, dupes))}")
    absent = [s for s in sources if s not in df.columns]
    if absent:
        raise ValueError(f"Mapped column(s) not found in the file: {', '.join(map(str, absent))}")
    out = df[sources].copy()
    out.columns = list(chosen)
    return out
