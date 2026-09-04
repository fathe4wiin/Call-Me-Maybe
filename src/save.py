"""Write the output JSON array of function calls."""

from __future__ import annotations

import json
from pathlib import Path

from .models import OutputRecord


def save_results(path: Path, records: list[OutputRecord]) -> None:
    """Write *records* as a JSON array, creating parent directories.

    Args:
        path: Destination given by ``--output``.
        records: One object per input prompt.

    Raises:
        OSError: If the file cannot be created or written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump() for record in records]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
