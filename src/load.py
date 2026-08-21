"""Load and validate the input JSON files. Fail with a clear message, no traceback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn, TypeVar

from pydantic import BaseModel, ValidationError

from .models import FunctionDefinition, TestPrompt

TModel = TypeVar("TModel", bound=BaseModel)


class JsonLoader(BaseModel):
    """Takes a path, not an open file. Open inside load_raw() with a context manager."""

    path: Path

    def _fail(self, reason: str) -> NoReturn:
        """Print one line on stderr and stop. Never used for successful loads."""
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

    def _load_list(self, item_model: type[TModel], label: str) -> list[TModel]:
        raw = self.load_raw()
        if not isinstance(raw, list):
            self._fail(f"expected a JSON array of {label}, got {type(raw).__name__}")
        try:
            return [item_model.model_validate(item) for item in raw]
        except ValidationError as exc:
            self._fail(f"does not match the {label} schema ({exc.error_count()} issue(s))")

    def load_functions(self) -> list[FunctionDefinition]:
        """Validate the file as a list of function definitions."""
        return self._load_list(FunctionDefinition, "function definitions")

    def load_prompts(self) -> list[TestPrompt]:
        """Validate the file as a list of test prompts."""
        return self._load_list(TestPrompt, "prompts")
