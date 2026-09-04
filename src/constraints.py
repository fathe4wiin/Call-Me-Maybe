"""Character-level JSON + schema constraint machine."""

from __future__ import annotations

from enum import Enum
from typing import assert_never

from pydantic import BaseModel

from .models import FunctionDefinition, ParamSchema
from .prompt import JSON_PREFIX

_AFTER_NAME = ',"parameters":{'
_MAX_STRING = 200
_MAX_NUMBER = 24
_ESCAPE_CHARS = set('"\\/bfnrt')
_HEX = set("0123456789abcdefABCDEF")
_SNAPSHOT_FIELDS = (
    "phase",
    "literal_rest",
    "after_literal",
    "name_buffer",
    "chosen_index",
    "param_index",
    "kind",
    "value_buffer",
    "string_open",
    "string_escape",
    "unicode_left",
    "number_digits",
    "number_dot",
    "number_frac",
    "bool_buffer",
)


class Phase(str, Enum):
    """Where we are in the forced JSON template."""

    NAME = "name"
    LITERAL = "literal"
    VALUE = "value"
    DONE = "done"


class ParamKind(str, Enum):
    """Normalized parameter types from the catalog."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class AfterLiteral(str, Enum):
    """What to do when the current literal string is fully consumed."""

    FIRST_PARAM = "first_param"
    START_VALUE = "start_value"
    DONE = "done"


def normalize_kind(raw: str) -> ParamKind:
    """Map a catalog type string onto the kinds we constrain.

    Args:
        raw: The ``type`` field from the function definition.

    Returns:
        A ``ParamKind`` used by the value decoder.
    """
    lowered = raw.strip().lower()
    if lowered in ("string", "str"):
        return ParamKind.STRING
    if lowered in ("number", "float", "double"):
        return ParamKind.NUMBER
    if lowered in ("integer", "int"):
        return ParamKind.INTEGER
    if lowered in ("boolean", "bool"):
        return ParamKind.BOOLEAN
    return ParamKind.STRING


class DecodeState(BaseModel):
    """Mutable constrained-decoding state for one function call."""

    functions: list[FunctionDefinition]
    phase: Phase = Phase.NAME
    generated: str = JSON_PREFIX
    literal_rest: str = ""
    after_literal: AfterLiteral = AfterLiteral.FIRST_PARAM
    name_buffer: str = ""
    chosen_index: int | None = None
    param_index: int = 0
    kind: ParamKind = ParamKind.STRING
    value_buffer: str = ""
    string_open: bool = False
    string_escape: bool = False
    unicode_left: int = 0
    number_digits: int = 0
    number_dot: bool = False
    number_frac: int = 0
    bool_buffer: str = ""

    @classmethod
    def start(cls, functions: list[FunctionDefinition]) -> DecodeState:
        """Build a machine that has already emitted ``{"name":"``."""
        return cls(functions=functions)

    def is_complete(self) -> bool:
        """Return True when the JSON object is closed."""
        return self.phase is Phase.DONE

    def chosen_name(self) -> str:
        """Return the catalog name selected during the name phase.

        Raises:
            RuntimeError: If the function name has not been chosen yet.
        """
        return self._chosen().name

    def accepts(self, text: str) -> bool:
        """Return True if *text* can be appended without breaking the schema."""
        if not text:
            return False
        snap = self._snapshot()
        ok = True
        for char in text:
            if not self.feed_char(char, record=False):
                ok = False
                break
        self._restore(snap)
        return ok

    def advance(self, text: str) -> None:
        """Commit *text*. Caller must have checked ``accepts``."""
        for char in text:
            if not self.feed_char(char, record=True):
                raise RuntimeError("advance() received an illegal token")

    def feed_char(self, char: str, record: bool) -> bool:
        """Consume one character. Returns False if it is illegal right now."""
        phase = self.phase
        if phase is Phase.NAME:
            ok = self._feed_name(char)
        elif phase is Phase.LITERAL:
            ok = self._feed_literal(char)
        elif phase is Phase.VALUE:
            ok = self._feed_value(char)
        elif phase is Phase.DONE:
            return False
        else:
            assert_never(phase)
        if ok and record:
            self.generated += char
        return ok

    def allowed_first_chars(self) -> set[str] | None:
        """First characters that can still be legal.

        Returns:
            A set of characters, or ``None`` if a string body is open
            (almost any character might appear).
        """
        phase = self.phase
        if phase is Phase.DONE:
            return set()
        if phase is Phase.LITERAL:
            if not self.literal_rest:
                return set()
            return {self.literal_rest[0]}
        if phase is Phase.NAME:
            chars: set[str] = set()
            for function in self.functions:
                name = function.name
                if not name.startswith(self.name_buffer):
                    continue
                rest = name[len(self.name_buffer):]
                if rest:
                    chars.add(rest[0])
                else:
                    chars.add('"')
            return chars
        if phase is Phase.VALUE:
            return self._value_first_chars()
        assert_never(phase)

    def _value_first_chars(self) -> set[str] | None:
        """Allowed first characters while filling a parameter value."""
        kind = self.kind
        if kind is ParamKind.STRING:
            if not self.string_open:
                return {'"'}
            if self.unicode_left:
                return set(_HEX)
            if self.string_escape:
                return set(_ESCAPE_CHARS) | set("u")
            if len(self.value_buffer) >= _MAX_STRING:
                return {'"'}
            return None
        if kind is ParamKind.BOOLEAN:
            allowed: set[str] = set()
            for word in ("true", "false"):
                if word.startswith(self.bool_buffer):
                    rest = word[len(self.bool_buffer):]
                    if rest:
                        allowed.add(rest[0])
                    else:
                        allowed.update(self._value_terminators())
            return allowed
        if kind is ParamKind.NUMBER or kind is ParamKind.INTEGER:
            return self._number_first_chars(kind)
        assert_never(kind)

    def _number_first_chars(self, kind: ParamKind) -> set[str]:
        """Allowed first characters for a JSON number / integer."""
        buf = self.value_buffer
        chars: set[str] = set()
        if not buf:
            chars.update("-0123456789")
            return chars
        if buf == "-":
            return set("0123456789")
        if self._number_complete():
            chars.update(self._value_terminators())
        if len(buf) >= _MAX_NUMBER:
            return chars or self._value_terminators()
        last = buf[-1]
        if last == ".":
            return set("0123456789")
        if self.number_digits:
            body = buf.lstrip("-")
            if not (body.startswith("0") and not self.number_dot):
                chars.update("0123456789")
            if (
                kind is ParamKind.NUMBER
                and not self.number_dot
                and last.isdigit()
            ):
                chars.add(".")
        return chars

    def _feed_name(self, char: str) -> bool:
        """Accept a character of the function name, or the closing quote."""
        if char == '"':
            for index, function in enumerate(self.functions):
                if function.name == self.name_buffer:
                    self.chosen_index = index
                    self.phase = Phase.LITERAL
                    self.literal_rest = _AFTER_NAME
                    self.after_literal = AfterLiteral.FIRST_PARAM
                    return True
            return False
        nxt = self.name_buffer + char
        if any(function.name.startswith(nxt) for function in self.functions):
            self.name_buffer = nxt
            return True
        return False

    def _feed_literal(self, char: str) -> bool:
        """Accept the next character of a forced structural string."""
        if not self.literal_rest or self.literal_rest[0] != char:
            return False
        self.literal_rest = self.literal_rest[1:]
        if not self.literal_rest:
            self._finish_literal()
        return True

    def _finish_literal(self) -> None:
        """Move to the next phase after a forced fragment is done."""
        action = self.after_literal
        if action is AfterLiteral.FIRST_PARAM:
            self.param_index = 0
            self._begin_param_or_close()
            return
        if action is AfterLiteral.START_VALUE:
            self._begin_value()
            return
        if action is AfterLiteral.DONE:
            self.phase = Phase.DONE
            return
        assert_never(action)

    def _begin_param_or_close(self) -> None:
        """Emit the next ``\"key\":`` literal, or close the JSON object."""
        function = self._chosen()
        keys = list(function.parameters.keys())
        if self.param_index >= len(keys):
            self.phase = Phase.LITERAL
            self.literal_rest = "}}"
            self.after_literal = AfterLiteral.DONE
            return
        key = keys[self.param_index]
        comma = "" if self.param_index == 0 else ","
        self.phase = Phase.LITERAL
        self.literal_rest = f'{comma}"{key}":'
        self.after_literal = AfterLiteral.START_VALUE

    def _begin_value(self) -> None:
        """Start constraining the current parameter's value."""
        function = self._chosen()
        keys = list(function.parameters.keys())
        schema: ParamSchema = function.parameters[keys[self.param_index]]
        self.kind = normalize_kind(schema.type)
        self.phase = Phase.VALUE
        self.value_buffer = ""
        self.string_open = False
        self.string_escape = False
        self.unicode_left = 0
        self.number_digits = 0
        self.number_dot = False
        self.number_frac = 0
        self.bool_buffer = ""

    def _feed_value(self, char: str) -> bool:
        """Dispatch a value character to the active parameter type."""
        kind = self.kind
        if kind is ParamKind.STRING:
            return self._feed_string(char)
        if kind is ParamKind.NUMBER or kind is ParamKind.INTEGER:
            return self._feed_number(char, kind)
        if kind is ParamKind.BOOLEAN:
            return self._feed_bool(char)
        assert_never(kind)

    def _feed_string(self, char: str) -> bool:
        """Accept one character of a JSON string value."""
        if not self.string_open:
            if char != '"':
                return False
            self.string_open = True
            return True
        if self.unicode_left:
            if char not in _HEX:
                return False
            self.unicode_left -= 1
            self.value_buffer += char
            return True
        if self.string_escape:
            if char == "u":
                self.unicode_left = 4
                self.string_escape = False
                self.value_buffer += char
                return True
            if char not in _ESCAPE_CHARS:
                return False
            self.string_escape = False
            self.value_buffer += char
            return True
        if char == "\\":
            self.string_escape = True
            self.value_buffer += char
            return True
        if char == '"':
            self._commit_value()
            return True
        if ord(char) < 0x20:
            return False
        if len(self.value_buffer) >= _MAX_STRING:
            return False
        self.value_buffer += char
        return True

    def _feed_number(self, char: str, kind: ParamKind) -> bool:
        """Accept one character of a JSON number, or a terminator."""
        if char in self._value_terminators() and self._number_complete():
            self._commit_value()
            return self.feed_char(char, record=False)
        buf = self.value_buffer
        if char == "-" and not buf:
            self.value_buffer = "-"
            return True
        if char == ".":
            if kind is ParamKind.INTEGER:
                return False
            if self.number_dot or not self.number_digits:
                return False
            self.number_dot = True
            self.value_buffer += char
            return True
        if char.isdigit():
            if len(buf) >= _MAX_NUMBER:
                return False
            body = buf.lstrip("-")
            if body == "0" and not self.number_dot:
                return False
            if self.number_dot:
                self.number_frac += 1
            else:
                self.number_digits += 1
            self.value_buffer += char
            return True
        return False

    def _feed_bool(self, char: str) -> bool:
        """Accept the next character of ``true`` / ``false``, or a terminator."""
        if char in self._value_terminators() and self.bool_buffer in (
            "true",
            "false",
        ):
            self._commit_value()
            return self.feed_char(char, record=False)
        nxt = self.bool_buffer + char
        if "true".startswith(nxt) or "false".startswith(nxt):
            self.bool_buffer = nxt
            return True
        return False

    def _number_complete(self) -> bool:
        """Return True if the buffered number is a valid JSON number so far."""
        if self.number_digits < 1:
            return False
        if self.number_dot and self.number_frac < 1:
            return False
        return True

    def _value_terminators(self) -> set[str]:
        """Characters that may follow a finished number or boolean."""
        function = self._chosen()
        last = self.param_index >= len(function.parameters) - 1
        return {"}"} if last else {","}

    def _commit_value(self) -> None:
        """Current parameter is done; move to the next key or the closer."""
        self.param_index += 1
        self._begin_param_or_close()

    def _chosen(self) -> FunctionDefinition:
        """Return the function selected during the name phase."""
        if self.chosen_index is None:
            raise RuntimeError("function name has not been chosen yet")
        return self.functions[self.chosen_index]

    def _snapshot(self) -> dict[str, object]:
        """Copy the mutable fields used by ``accepts``."""
        return {name: getattr(self, name) for name in _SNAPSHOT_FIELDS}

    def _restore(self, snap: dict[str, object]) -> None:
        """Restore fields saved by ``_snapshot``."""
        for name, value in snap.items():
            setattr(self, name, value)
