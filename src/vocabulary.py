"""Map tokenizer vocabulary IDs to the text they actually insert."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


def _bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 / Qwen byte-level BPE printable mapping.

    Vocab files store a leading space as ``Ġ``, not as a real space.
    Reversing this table recovers the text a token will append.
    """
    latin = list(range(ord("!"), ord("~") + 1))
    latin += list(range(ord("¡"), ord("¬") + 1))
    latin += list(range(ord("®"), ord("ÿ") + 1))
    codes = list(latin)
    extra = 0
    for byte in range(256):
        if byte not in latin:
            latin.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(latin, [chr(code) for code in codes]))


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {char: byte for byte, char in _BYTE_TO_UNICODE.items()}


def token_piece_to_text(piece: str) -> str:
    """Decode one vocab piece into the string it represents.

    Args:
        piece: Token string as stored in ``vocab.json``.

    Returns:
        UTF-8 text, or an empty string if the piece is not valid UTF-8.
    """
    try:
        raw = bytes(_UNICODE_TO_BYTE[char] for char in piece)
    except KeyError:
        return piece
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def _extract_pieces(payload: object) -> dict[str, int]:
    """Return a token-piece -> id map from either vocab file format."""
    if not isinstance(payload, dict):
        raise ValueError("vocabulary file must be a JSON object")
    model = payload.get("model")
    if isinstance(model, dict):
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            raise ValueError("tokenizer.json is missing model.vocab")
        pieces = {str(key): int(value) for key, value in vocab.items()}
        added = payload.get("added_tokens", [])
        if isinstance(added, list):
            for item in added:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                token_id = item.get("id")
                if isinstance(content, str) and isinstance(token_id, int):
                    pieces[content] = token_id
        return pieces
    return {str(key): int(value) for key, value in payload.items()}


class Vocabulary(BaseModel):
    """Token-id to text table built from the SDK vocab / tokenizer file."""

    id_to_text: dict[int, str] = Field(default_factory=dict)
    by_first: dict[str, list[int]] = Field(default_factory=dict)
    all_ids: list[int] = Field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> Vocabulary:
        """Load ``vocab.json`` or ``tokenizer.json`` from *path*."""
        file_path = Path(path)
        with file_path.open(encoding="utf-8") as handle:
            payload: object = json.load(handle)
        pieces = _extract_pieces(payload)
        id_to_text: dict[int, str] = {}
        for piece, token_id in pieces.items():
            text = token_piece_to_text(piece)
            if text:
                id_to_text[int(token_id)] = text
        by_first: dict[str, list[int]] = {}
        for token_id, text in id_to_text.items():
            by_first.setdefault(text[0], []).append(token_id)
        return cls(
            id_to_text=id_to_text,
            by_first=by_first,
            all_ids=list(id_to_text.keys()),
        )

    def text_of(self, token_id: int) -> str:
        """Return the text for *token_id*, or empty if unknown."""
        return self.id_to_text.get(token_id, "")

    def candidate_ids(self, first_chars: set[str] | None) -> list[int]:
        """Token IDs whose text starts with one of *first_chars*.

        Args:
            first_chars: Allowed first characters, or ``None`` to scan all
                (used while a JSON string body is open).

        Returns:
            Token IDs to test with the constraint machine.
        """
        if first_chars is None:
            return self.all_ids
        out: list[int] = []
        for char in first_chars:
            out.extend(self.by_first.get(char, []))
        return out
