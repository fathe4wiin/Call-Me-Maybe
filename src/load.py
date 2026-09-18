"""Load and validate input JSON. Fail with one message, no traceback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn, TypeVar

from pydantic import BaseModel, ValidationError

from .models import FunctionDefinition, TestPrompt

TModel = TypeVar("TModel", bound=BaseModel)


def _is_empty_string(value: object) -> bool:
    """Return True only for the JSON string ``""``."""
    return isinstance(value, str) and value == ""


def _empty_string_path(
    value: object,
    prefix: str = "",
    *,
    allow_empty: frozenset[str] | None = None,
) -> str | None:
    """Return a dotted path to the first empty string key or value.

    Args:
        value: Decoded JSON value.
        prefix: Path accumulated from parent objects and arrays.
        allow_empty: Field names whose empty string values are allowed
            (``prompt`` in the tests file).

    Returns:
        A location such as ``[0].parameters.a.type``, or ``None``.
    """
    allowed = allow_empty or frozenset()
    if _is_empty_string(value):
        name = prefix.rsplit(".", 1)[-1]
        if name in allowed:
            return None
        return prefix or "(root)"
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_empty_string(key):
                return f"{prefix}['']" if prefix else "['']"
            child = f"{prefix}.{key}" if prefix else str(key)
            found = _empty_string_path(
                item, child, allow_empty=allowed
            )
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            found = _empty_string_path(
                item, child, allow_empty=allowed
            )
            if found is not None:
                return found
        return None
    return None


class JsonLoader(BaseModel):
    """Takes a path. Open it inside load_raw() with a context manager."""

    path: Path

    def _fail(self, reason: str) -> NoReturn:
        """Print one stderr line and stop. Unused on successful loads."""
        print(f"error: {self.path}: {reason}", file=sys.stderr)
        sys.exit(1)

    def load_raw(self) -> object:
        """Open the path and return the decoded JSON value."""
        try:
            with self.path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            self._fail("file not found")
        except IsADirectoryError:
            self._fail("expected a file, got a directory")
        except OSError as exc:
            self._fail(f"cannot read file ({exc.strerror})")
        except json.JSONDecodeError as exc:
            self._fail(f"invalid JSON ({exc.msg} at line {exc.lineno})")
        except Exception as exc:
            self._fail(f"unexpected error ({exc.__class__.__name__}: {exc})")

    def _load_list(
        self,
        item_model: type[TModel],
        label: str,
        *,
        allow_empty: frozenset[str] | None = None,
    ) -> list[TModel]:
        """Validate a JSON array of *item_model* objects.

        Args:
            item_model: Pydantic class for one array element.
            label: Human-readable name used in error messages.
            allow_empty: Field names that may be ``""`` (tests: ``prompt``).

        Returns:
            One validated model per array element.
        """
        raw = self.load_raw()
        if not isinstance(raw, list):
            self._fail(
                f"expected a JSON array of {label}, got {type(raw).__name__}"
            )
        found = _empty_string_path(raw, allow_empty=allow_empty)
        if found is not None:
            self._fail(f"empty string at {found}")
        try:
            return [item_model.model_validate(item) for item in raw]
        except ValidationError as exc:
            count = exc.error_count()
            self._fail(f"does not match the {label} schema ({count} issue(s))")

    def load_functions(self) -> list[FunctionDefinition]:
        """Validate the file as a list of function definitions."""
        return self._load_list(FunctionDefinition, "function definitions")

    def load_prompts(self) -> list[TestPrompt]:
        """Validate the file as a list of test prompts.

        An empty ``prompt`` is kept. Generation maps it to ``unknown``.
        """
        return self._load_list(
            TestPrompt,
            "prompts",
            allow_empty=frozenset({"prompt"}),
        )
